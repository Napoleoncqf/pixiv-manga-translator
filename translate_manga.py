"""
漫画翻译脚本 - 日文 → 中文

完整管线：检测 → OCR → 颜色采样 → 过滤 → 翻译 → 修复 → 渲染 → 审查
- 圆润可爱字体（站酷快乐体）
- 保留原文颜色
- 过滤音效字和浮动标注
- 自动扩展窄检测框
- Gemini 视觉审查：检测残留日文/未擦除符号，不合格自动标记

使用方法:
    1. 确保 Saber-Translator 后端运行中 (python app.py)
    2. 在 .env 中配置 GEMINI_API_KEY
    3. 运行: python translate_manga.py [输入目录] [输出目录]
       默认: python translate_manga.py 3sisters 3sisters/translated
"""
import requests
import json
import base64
import os
import sys
import re
import io
import argparse
import numpy as np
from PIL import Image, ImageDraw
from dotenv import load_dotenv

# 加载 .env 配置
load_dotenv()

API_BASE = os.getenv("TRANSLATE_API_BASE", "http://127.0.0.1:5000/api")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("TRANSLATE_MODEL", "gemini-3.1-pro-preview")
FONT = os.getenv("TRANSLATE_FONT", "src/app/static/fonts/ZhanKuKuaiLeTi.ttf")
TARGET_LANG = os.getenv("TRANSLATE_TARGET_LANG", "Chinese")
SOURCE_LANG = os.getenv("TRANSLATE_SOURCE_LANG", "japanese")


# ── 工具函数 ─────────────────────────────────────────────

def extract_text_color(img_arr, x1, y1, x2, y2):
    """从像素采样文字颜色：取气泡区域暗像素的中位色。"""
    region = img_arr[y1:y2, x1:x2]
    if region.size == 0:
        return "#000000", None
    gray = np.mean(region, axis=2)
    text_mask = gray < 120
    if text_mask.sum() < 10:
        text_mask = gray < 160
    if text_mask.sum() < 5:
        return "#000000", None
    text_pixels = region[text_mask]
    r, g, b = (text_pixels[:, c].astype(float) for c in range(3))
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = np.where(max_c > 0, (max_c - min_c) / max_c, 0)
    colored_mask = sat > 0.10
    pixels = text_pixels[colored_mask] if colored_mask.sum() > 10 else text_pixels
    med = np.median(pixels, axis=0).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*med), tuple(int(x) for x in med)


def img_to_b64(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(ext, ext)
    return f"data:image/{mime};base64,{data}"


def b64_to_file(b64_str, path):
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_str))


def is_bubble_text(coord, text):
    """判断是否为气泡内对话文字。
    过滤：无日文内容、极扁横条（时间戳）、极窄短文本（音效字）。
    """
    x1, y1, x2, y2 = coord
    w, h = x2 - x1, y2 - y1
    if not re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text):
        return False
    if w / max(h, 1) > 4 and h < 80:
        return False
    if w < 75 and len(text) <= 4:
        return False
    return True


def paint_white_patches(clean_img_bytes, bubble_coords, scale=1.15):
    """对 clean image 的每个气泡区域画一层白色圆角矩形（兜底方案）。

    用于浮动文字/复杂背景场景 - LAMA 擦不干净时，直接覆盖为白底，
    后续渲染中文就不会看到任何日文残留。

    Args:
        clean_img_bytes: PNG bytes of the clean (already inpainted) image
        bubble_coords: list of [x1,y1,x2,y2] for bubbles to patch
        scale: how much to expand each bubble before painting (1.0 = exact)

    Returns:
        PNG bytes of the patched image
    """
    img = Image.open(io.BytesIO(clean_img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    for c in bubble_coords:
        x1, y1, x2, y2 = c
        cx, cy = (x1+x2)/2, (y1+y2)/2
        hw, hh = (x2-x1)/2 * scale, (y2-y1)/2 * scale
        px1, py1 = max(0, int(cx-hw)), max(0, int(cy-hh))
        px2, py2 = min(iw, int(cx+hw)), min(ih, int(cy+hh))
        # 圆角矩形白底
        radius = max(10, int(min(px2-px1, py2-py1) * 0.15))
        try:
            draw.rounded_rectangle([px1, py1, px2, py2], radius=radius, fill=(255, 255, 255))
        except AttributeError:
            # 老版 Pillow 没有 rounded_rectangle
            draw.rectangle([px1, py1, px2, py2], fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def expand_narrow_coords(coords, directions, img_w, img_h):
    """竖排窄框扩展：宽度 < 130px 时左右扩展到至少 140px。"""
    expanded = []
    for i, c in enumerate(coords):
        x1, y1, x2, y2 = c
        w = x2 - x1
        d = directions[i] if i < len(directions) else "v"
        if d == "v" and w < 130:
            target_w = max(140, int(w * 1.8))
            pad = (target_w - w) // 2
            x1 = max(0, x1 - pad)
            x2 = min(img_w, x2 + pad)
        expanded.append([x1, y1, x2, y2])
    return expanded


# ── 审查函数 ─────────────────────────────────────────────

def review_translated_image(original_path, translated_path):
    """用 Gemini 视觉审查翻译后的图片（对比原图）。

    同时发送原图和翻译图，让模型对比检查修复质量。

    Returns:
        (passed: bool, issues: list[str])
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=GEMINI_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    def read_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def get_mime(path):
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        return {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "png")

    orig_b64 = read_b64(original_path)
    trans_b64 = read_b64(translated_path)

    response = client.chat.completions.create(
        model="gemini-3.1-pro-preview",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "你是漫画翻译质量审查员。\n\n"
                        "第一张是日文原图，第二张是翻译后的中文图。\n"
                        "请逐个检查翻译图中的每个气泡，判断是否合格。\n\n"
                        "**只有以下情况才判不合格（false）：**\n"
                        "1. 气泡内能清晰辨认出**整个或半个日文字符**（能读出是什么字/假名）\n"
                        "2. 原日文和中文译文明显并排或叠加在一起，两种文字都很清晰\n"
                        "3. 原文旁有清晰完整的装饰符号（大大的 ♡♪★）没擦掉\n"
                        "4. 中文译文本身仍然是日文（翻译失败，没翻译成中文）\n\n"
                        "**不算问题（必须忽略）：**\n"
                        "- 擦除区域有轻微的灰色痕迹、模糊色块（JPEG 压缩的正常现象）\n"
                        "- 中文笔画周围有细微的像素噪点\n"
                        "- 气泡背景不是纯白色而是浅灰/浅粉（漫画原本就有底色）\n"
                        "- 气泡外的音效字、时间标记（AM 0:30 等）\n"
                        "- 人物/画面本身的细节、阴影\n\n"
                        "**判定标准：能否清楚读出是什么日文字**——如果只是模糊痕迹但认不出是什么字，就不算问题。\n\n"
                        "回复严格 JSON：\n"
                        '{"passed": true/false, "issues": ["气泡位置: 具体问题（说明你看到了什么日文字）"]}\n'
                        "严禁因为 JPEG 压缩痕迹而判不合格。"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{get_mime(original_path)};base64,{orig_b64}"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{get_mime(translated_path)};base64,{trans_b64}"}
                }
            ]
        }],
        temperature=0.1,
        max_tokens=2000
    )

    raw = response.choices[0].message.content.strip()
    finish_reason = response.choices[0].finish_reason

    # 提取 JSON
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return result.get("passed", True), result.get("issues", [])
        except json.JSONDecodeError:
            pass

    # JSON 解析失败时，从文本内容判断
    # 如果回复被截断且提到了问题，默认不通过
    raw_lower = raw.lower()
    if finish_reason == "length":
        # 截断了，看已有内容是否提到问题
        if any(kw in raw for kw in ["残留", "没擦", "重叠", "痕迹", "false", "issues"]):
            return False, [f"审查回复被截断但提及问题: {raw[:200]}"]
    if "passed" in raw_lower and "false" in raw_lower:
        return False, [f"从文本判断不通过: {raw[:200]}"]

    return True, []


# ── 主流程 ───────────────────────────────────────────────

def translate_image(img_path, output_dir):
    name = os.path.basename(img_path)
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    with open(img_path, "rb") as f:
        raw = f.read()
    img_pil = Image.open(io.BytesIO(raw)).convert("RGB")
    img_arr = np.array(img_pil)
    img_w, img_h = img_pil.size
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(ext, ext)
    img_b64 = f"data:image/{mime};base64," + base64.b64encode(raw).decode()

    # Step 1: 检测
    print("[1/6] Detecting...")
    det = requests.post(f"{API_BASE}/parallel/detect",
        json={"image": img_b64, "detector_type": "ctd"}, timeout=120).json()
    if not det.get("success"):
        print(f"  FAILED: {det.get('error')}"); return
    all_coords = det["bubble_coords"]
    all_polygons = det.get("bubble_polygons", [])
    all_directions = det.get("auto_directions", [])
    raw_mask = det.get("raw_mask")
    if not all_coords:
        print("  No text found."); return

    # Step 2: OCR
    print("[2/6] OCR...")
    ocr = requests.post(f"{API_BASE}/parallel/ocr",
        json={"image": img_b64, "bubble_coords": all_coords, "source_language": SOURCE_LANG},
        timeout=120).json()
    if not ocr.get("success"):
        print(f"  FAILED: {ocr.get('error')}"); return
    all_texts = ocr.get("original_texts", [])

    # Step 3: 颜色采样 + 过滤
    print("[3/6] Color & filter...")
    coords, polygons, directions, texts, text_colors, fg_rgbs = [], [], [], [], [], []
    for i in range(len(all_coords)):
        t = all_texts[i] if i < len(all_texts) else ""
        c = all_coords[i]
        w, h = c[2]-c[0], c[3]-c[1]
        if not is_bubble_text(c, t):
            print(f"  ✗ [{i}] \"{t}\"  ({w}x{h}, filtered)")
            continue
        hx, rgb = extract_text_color(img_arr, *c)
        coords.append(c)
        polygons.append(all_polygons[i] if i < len(all_polygons) else [])
        directions.append(all_directions[i] if i < len(all_directions) else "v")
        texts.append(t)
        text_colors.append(hx)
        fg_rgbs.append(rgb)
        print(f"  ✓ [{i}] \"{t}\"  ({w}x{h}, color={hx})")

    if not coords:
        print("  No bubble text to translate."); return

    # 扩展窄框
    render_coords = expand_narrow_coords(coords, directions, img_w, img_h)
    for i in range(len(coords)):
        if coords[i] != render_coords[i]:
            ow = coords[i][2] - coords[i][0]
            nw = render_coords[i][2] - render_coords[i][0]
            print(f"  ↔ Expanded [{i}] width {ow} → {nw}")

    # Step 4: 翻译
    def do_translate(model_name):
        return requests.post(f"{API_BASE}/parallel/translate",
            json={
                "original_texts": texts, "target_language": TARGET_LANG,
                "source_language": SOURCE_LANG, "model_provider": "gemini",
                "model_name": model_name, "api_key": GEMINI_KEY, "rpm_limit": 60
            }, timeout=180).json()

    def has_jp_kana(s):
        return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF]', s))

    print(f"[4/6] Translating ({MODEL})...")
    tr = do_translate(MODEL)
    if not tr.get("success"):
        print(f"  FAILED: {tr.get('error')}"); return
    translated = tr.get("translated_texts", [])

    # 检测翻译失败的条目（译文仍大量含日文假名）并单独重译
    for i, t in enumerate(translated):
        if has_jp_kana(t):
            print(f"  ⚠ [{i+1}] 翻译失败仍含假名: {t}")
            # 单条重译
            retry = requests.post(f"{API_BASE}/translate_single_text",
                json={
                    "original_text": texts[i],
                    "target_language": TARGET_LANG,
                    "api_key": GEMINI_KEY,
                    "model_name": MODEL,
                    "model_provider": "gemini",
                    "prompt_content": (
                        "你是漫画翻译。把以下日文翻译成自然的中文。"
                        "如果是拟声词/尖叫声（うわああ、きゃあ等），翻译为对应的中文拟声（呜哇啊啊、啊——等）。"
                        "只输出翻译结果，不要加任何说明。"
                    )
                }, timeout=60).json()
            new_t = retry.get("translated_text", "")
            if new_t and not has_jp_kana(new_t):
                translated[i] = new_t
                print(f"     ↻ 重译成功: {new_t}")

    for i, (o, t) in enumerate(zip(texts, translated)):
        print(f"  [{i+1}] {o} => {t}")

    # Step 5-7: 修复 → 渲染 → 审查（不合格自动重试，逐轮加大修复力度）
    # 精确掩膜在 JPEG 图上不可靠（笔画边缘漏检），默认就用整框擦除
    # 重试策略：
    #   轮1: 整框擦除 + dilate=10 + lama_mpe
    #   轮2: 整框扩展1.3倍 + dilate=25 + lama_mpe
    #   轮3: 整框扩展1.6倍 + dilate=50 + litelama  — 暴力模式
    RETRY_CONFIGS = [
        {"dilate": 10, "model": "lama_mpe",  "use_raw_mask": False, "box_scale": 1.0, "label": "默认整框擦除"},
        {"dilate": 25, "model": "lama_mpe",  "use_raw_mask": False, "box_scale": 1.3, "label": "框扩1.3x+加大擦除"},
        {"dilate": 50, "model": "litelama",  "use_raw_mask": False, "box_scale": 1.6, "label": "框扩1.6x+换模型+最大擦除"},
    ]
    MAX_ATTEMPTS = len(RETRY_CONFIGS)

    out_path = os.path.join(output_dir, os.path.splitext(name)[0] + "_translated.png")

    for attempt, cfg in enumerate(RETRY_CONFIGS):
        attempt_label = f"[尝试 {attempt+1}/{MAX_ATTEMPTS}: {cfg['label']}]"

        # Inpaint — 用扩展坐标擦除，确保窄框文字不残留
        inpaint_coords = expand_narrow_coords(coords, directions, img_w, img_h)
        # 根据重试级别对所有 inpaint 坐标做整体放大
        scale = cfg.get("box_scale", 1.0)
        if scale > 1.0:
            scaled = []
            for c in inpaint_coords:
                x1, y1, x2, y2 = c
                cx, cy = (x1+x2)/2, (y1+y2)/2
                hw, hh = (x2-x1)/2 * scale, (y2-y1)/2 * scale
                scaled.append([
                    max(0, int(cx-hw)), max(0, int(cy-hh)),
                    min(img_w, int(cx+hw)), min(img_h, int(cy+hh))
                ])
            inpaint_coords = scaled

        print(f"[5/7] Inpainting {attempt_label}...")
        inp_data = {
            "image": img_b64, "bubble_coords": inpaint_coords, "bubble_polygons": polygons,
            "method": "lama", "lama_model": cfg["model"],
            "mask_dilate_size": cfg["dilate"]
        }
        # 只有第一轮用精确掩膜；重试时去掉，改用整框擦除更彻底
        if cfg["use_raw_mask"] and raw_mask:
            inp_data["raw_mask"] = raw_mask
        inp = requests.post(f"{API_BASE}/parallel/inpaint", json=inp_data, timeout=180).json()
        if not inp.get("success"):
            print(f"  FAILED: {inp.get('error')}"); return

        # Render
        print(f"[6/7] Rendering {attempt_label}...")
        bubble_states = []
        for i in range(len(coords)):
            td = "vertical" if directions[i] == "v" else "horizontal"
            bubble_states.append({
                "originalText": texts[i],
                "translatedText": translated[i] if i < len(translated) else "",
                "textboxText": "",
                "coords": render_coords[i],
                "polygon": polygons[i],
                "fontSize": 28,
                "fontFamily": FONT,
                "textDirection": td,
                "autoTextDirection": td,
                "textColor": text_colors[i],
                "fillColor": "#FFFFFF",
                "rotationAngle": 0,
                "position": {"x": 0, "y": 0},
                "strokeEnabled": False,
                "strokeColor": "#FFFFFF",
                "strokeWidth": 0,
                "inpaintMethod": "lama",
                "autoFgColor": list(fg_rgbs[i]) if fg_rgbs[i] else None,
                "autoBgColor": None
            })

        render = requests.post(f"{API_BASE}/parallel/render",
            json={
                "clean_image": inp["clean_image"],
                "bubble_states": bubble_states,
                "fontSize": 28,
                "fontFamily": FONT,
                "textDirection": "vertical",
                "textColor": "#000000",
                "strokeEnabled": False,
                "strokeColor": "#FFFFFF",
                "strokeWidth": 0,
                "autoFontSize": True,
                "use_individual_styles": True
            }, timeout=120).json()
        if not render.get("success"):
            print(f"  FAILED: {render.get('error')}"); return

        b64_to_file(render["final_image"], out_path)

        # Review
        print(f"[7/7] Reviewing {attempt_label}...")
        try:
            passed, issues = review_translated_image(img_path, out_path)
        except Exception as e:
            print(f"  ⚠ Review error: {e}"); passed = True; issues = []

        if passed:
            print(f"  ✓ PASSED (attempt {attempt+1})")
            break
        else:
            print(f"  ✗ ISSUES:")
            for issue in issues:
                print(f"    - {issue}")
            if attempt < MAX_ATTEMPTS - 1:
                next_cfg = RETRY_CONFIGS[attempt + 1]
                print(f"  → 自动重试: {next_cfg['label']} (dilate={next_cfg['dilate']}, model={next_cfg['model']})")
            else:
                # 兜底：在最激进的 clean image 上画白底覆盖问题气泡，再渲染一次
                print(f"  → 兜底方案: 对所有气泡加白底覆盖后重新渲染...")
                inpaint_coords_final = expand_narrow_coords(coords, directions, img_w, img_h)
                clean_bytes = base64.b64decode(inp["clean_image"].split(",", 1)[-1])
                patched_bytes = paint_white_patches(clean_bytes, inpaint_coords_final, scale=1.15)
                patched_b64 = "data:image/png;base64," + base64.b64encode(patched_bytes).decode()

                final_render = requests.post(f"{API_BASE}/parallel/render",
                    json={
                        "clean_image": patched_b64,
                        "bubble_states": bubble_states,
                        "fontSize": 28,
                        "fontFamily": FONT,
                        "textDirection": "vertical",
                        "textColor": "#000000",
                        "strokeEnabled": False,
                        "strokeColor": "#FFFFFF",
                        "strokeWidth": 0,
                        "autoFontSize": True,
                        "use_individual_styles": True
                    }, timeout=120).json()
                if final_render.get("success"):
                    b64_to_file(final_render["final_image"], out_path)
                    # 再审查一次
                    try:
                        passed2, issues2 = review_translated_image(img_path, out_path)
                    except Exception:
                        passed2 = True
                    if passed2:
                        print(f"  ✓ PASSED (兜底白底覆盖)")
                        break
                    else:
                        fail_path = out_path.replace("_translated.png", "_translated_REVIEW.png")
                        os.rename(out_path, fail_path)
                        print(f"  ✗ 兜底后仍不合格，标记人工复查")
                        for i in issues2:
                            print(f"    - {i}")
                else:
                    fail_path = out_path.replace("_translated.png", "_translated_REVIEW.png")
                    os.rename(out_path, fail_path)
                    print(f"  ✗ 兜底渲染失败")

    final_file = out_path if os.path.exists(out_path) else out_path.replace("_translated.png", "_translated_REVIEW.png")
    print(f"  Saved: {final_file}")


def main():
    parser = argparse.ArgumentParser(description="漫画翻译脚本 (日文→中文)")
    parser.add_argument("input_dir", nargs="?", default="3sisters", help="输入图片目录")
    parser.add_argument("output_dir", nargs="?", default=None, help="输出目录 (默认: 输入目录/translated)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "translated")
    os.makedirs(output_dir, exist_ok=True)

    if not GEMINI_KEY:
        print("ERROR: GEMINI_API_KEY not set. Add it to .env file.")
        sys.exit(1)

    image_exts = (".jpg", ".jpeg", ".png", ".webp")
    images = [f for f in sorted(os.listdir(input_dir))
              if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(image_exts)]

    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} image(s) in {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {MODEL}")
    print(f"Font: {FONT}")

    for fname in images:
        translate_image(os.path.join(input_dir, fname), output_dir)

    print(f"\n{'='*60}")
    print("All done!")


if __name__ == "__main__":
    main()
