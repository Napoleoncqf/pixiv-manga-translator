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
from PIL import Image, ImageDraw, ImageFont
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


# 角色分类标签（不翻译，保留原图）
ROLE_LABEL_WORDS = {
    "長女", "次女", "三女", "四女", "五女",
    "長姉", "次姉", "三姉",
    "長男", "次男", "三男",
    "兄", "姉", "妹", "弟",
    "お姉ちゃん", "お兄ちゃん",
    # 简体/繁体中文（以防 OCR 识别成汉字）
    "长女", "次女", "三女", "四女", "五女",
    "长姐", "次姐", "三姐",
}


def is_bubble_text(coord, text):
    """判断是否为气泡内对话文字。
    过滤：无日文内容、极扁横条（时间戳）、明确的音效字特征、角色分类标签词。
    """
    x1, y1, x2, y2 = coord
    w, h = x2 - x1, y2 - y1
    if not re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text):
        return False
    # 极扁横条（时间戳/水印）
    if w / max(h, 1) > 4 and h < 80:
        return False
    # 极窄且极短：只过滤宽 < 45px 且 ≤ 2 字符的明显音效字
    if w < 45 and len(text) <= 2:
        return False
    # 角色分类标签词（OCR 文本完全匹配标签集）
    cleaned = text.strip().strip("「」『』（）()[]【】").strip()
    if cleaned in ROLE_LABEL_WORDS:
        return False
    return True


def sanitize_translated_text(text):
    """清理译文中会导致中文字体渲染乱码的字符。

    站酷快乐体等纯中文字体不包含日文长音符 ー、全角波浪线 ～、
    装饰符号 ♡♪★ 等 —— 这些字符会被渲染成方块/豆腐字。
    统一替换为 Unicode 兼容的等效符号或删除。
    另外修复数字间被错用的中文全角句号（38。3 → 38.3）。
    """
    if not text:
        return text
    replacements = {
        "ー": "—",    # 日文长音符 → em dash
        "～": "~",    # 全角波浪线 → ASCII
        "♡": "♥",    # 空心爱心 → 实心
        "♪": "",
        "♬": "",
        "★": "",
        "☆": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 数字间的全角句号/逗号 → ASCII（38。3度 → 38.3度）
    text = re.sub(r'(\d)[。．](\d)', r'\1.\2', text)
    text = re.sub(r'(\d)，(\d)', r'\1,\2', text)

    # 去掉尾部所有非字母数字/中文/常用标点的垃圾符号
    text = re.sub(
        r'[^\w\u4e00-\u9fff\u3000-\u303f。，！？、：；""''「」『』（）—…·.,!?:;()\-~\s♥]+$',
        '', text
    )
    return text


def paint_white_patches(clean_img_bytes, bubble_coords, bubble_polygons=None, shrink=4):
    """在每个气泡内画白底（兜底方案，确保后续渲染时无残留）。

    优先使用 polygon（CTD 检测的实际气泡轮廓）填充，能贴合椭圆/不规则气泡形状。
    没有 polygon 的回退用 bbox 圆角矩形 + 内缩，避免遮挡气泡边线。

    Args:
        clean_img_bytes: PNG bytes
        bubble_coords: [[x1,y1,x2,y2], ...]
        bubble_polygons: [[[x,y], [x,y], ...], ...] 可选
        shrink: 内缩像素，避免覆盖气泡边线（默认 4）
    """
    img = Image.open(io.BytesIO(clean_img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    polygons = bubble_polygons or []

    for i, c in enumerate(bubble_coords):
        poly = polygons[i] if i < len(polygons) else None
        if poly and len(poly) >= 3:
            # 用 polygon 填充
            try:
                pts = [(int(p[0]), int(p[1])) for p in poly]
                draw.polygon(pts, fill=(255, 255, 255))
                continue
            except (TypeError, IndexError):
                pass

        # 回退：bbox 内缩 + 圆角矩形
        x1, y1, x2, y2 = c
        px1 = max(0, int(x1) + shrink)
        py1 = max(0, int(y1) + shrink)
        px2 = min(iw, int(x2) - shrink)
        py2 = min(ih, int(y2) - shrink)
        if px2 <= px1 or py2 <= py1:
            continue
        radius = max(8, int(min(px2-px1, py2-py1) * 0.2))
        try:
            draw.rounded_rectangle([px1, py1, px2, py2], radius=radius, fill=(255, 255, 255))
        except AttributeError:
            draw.rectangle([px1, py1, px2, py2], fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def sample_bubble_bg_color(img_arr, coord, polygon=None):
    """采样气泡的背景色（最亮的非文字像素中位色）。

    用于替代硬编码的白底 —— 当气泡在粉色/彩色背景上时能融合进去。
    对于常见的白色气泡仍然返回接近白色。
    """
    x1, y1, x2, y2 = [int(v) for v in coord]
    h, w = img_arr.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return (255, 255, 255)

    region = img_arr[y1:y2, x1:x2]
    gray = np.mean(region, axis=2)

    # 取区域里亮度前 30% 的像素作为"背景候选"（排除文字笔画）
    threshold = np.percentile(gray, 70)
    bg_mask = gray >= threshold
    if bg_mask.sum() < 20:
        # 回退：直接取全体像素中位
        bg_pixels = region.reshape(-1, 3)
    else:
        bg_pixels = region[bg_mask]

    median = np.median(bg_pixels, axis=0).astype(int)
    return tuple(int(x) for x in median)


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


# ── Vision 漏检补充 ─────────────────────────────────────────

def bbox_iou(a, b):
    """计算两个 bbox 的 IoU"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_missed_bubbles_via_vision(img_pil, existing_coords, max_dim=2400):
    """用 Gemini Vision 查漏补缺，返回 CTD 遗漏的气泡 bbox 列表（原图坐标）。

    使用较高分辨率（2400px）以捕获密集排列的小气泡。
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=GEMINI_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    img_w, img_h = img_pil.size
    scale = min(max_dim / img_w, max_dim / img_h, 1.0)
    if scale < 1.0:
        thumb = img_pil.resize((int(img_w * scale), int(img_h * scale)))
    else:
        thumb = img_pil
    tw, th = thumb.size

    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    existing_desc = ", ".join(
        f"[{int(c[0]*scale)},{int(c[1]*scale)},{int(c[2]*scale)},{int(c[3]*scale)}]"
        for c in existing_coords
    ) or "（无）"

    prompt = (
        f"这是一张日文漫画，图片尺寸 {tw}x{th}。\n"
        f"已检测到的文字区域 bbox（x1,y1,x2,y2）：{existing_desc}\n\n"
        f"**任务**：地毯式扫描这张图，找出**所有**包含日文文字的区域。\n"
        f"漫画常见漏检场景，请重点检查：\n\n"
        f"1. **密集场景里的小气泡**：第三/四格的多人场景中常有 5-10 个小气泡密集排列，\n"
        f"   每个只有 30-80 像素宽，但都需要单独检测。\n"
        f"2. **角色头顶/嘴边的小独白**：小写「…?」「えっ」「ふー」这类小独白，宽度可能只有 40px。\n"
        f"3. **画面背景里的旁白**：写在画面右侧/底部，无气泡边框的长段描述文字。\n"
        f"4. **大号粗体艺术字**：「スマホ禁止！」「頑張れ！」这种独占一格的装饰性大字。\n"
        f"5. **标题/章节名**：「長女」「次女」「三女」等角色标签（这些也要报告，由后续阶段决定如何处理）。\n"
        f"6. **复杂多格漫画**：每一格都要扫描，不要跳过任何分镜。\n\n"
        f"**不要包含**：\n"
        f"- 纯拟声词（ドキ、ガタ、キャ 等）\n"
        f"- 非文字的装饰符号\n"
        f"- 时间戳 AM 0:30\n"
        f"- 服装上的英文 Logo\n\n"
        f"**关键要求**：\n"
        f"- 已检测列表只是参考，新找到的区域只要与已有 bbox 重叠小于 30% 就要报告。\n"
        f"- 宁可多报不要漏报。多报的代价低，漏报会导致原日文残留在最终图上。\n"
        f"- 至少检查 3 遍才确认没有遗漏。\n\n"
        f"输出严格 JSON 数组（坐标必须是整数且在图片范围内）：\n"
        f"[[x1,y1,x2,y2], ...]\n"
        f"只输出 JSON，不要加任何说明。"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]}],
            temperature=0.1,
            max_tokens=800,
            timeout=120
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r'\[[\s\S]*\]', raw)
        if not m:
            return []
        result = json.loads(m.group())
        if not isinstance(result, list):
            return []
        # 缩放回原图坐标
        missed = []
        for c in result:
            if isinstance(c, list) and len(c) == 4:
                missed.append([
                    max(0, int(c[0] / scale)),
                    max(0, int(c[1] / scale)),
                    min(img_w, int(c[2] / scale)),
                    min(img_h, int(c[3] / scale)),
                ])
        return missed
    except Exception as e:
        print(f"  ⚠ Vision detection failed: {e}")
        return []


# ── Gemini 视觉翻译（B 方案：直接从气泡截图翻译，绕过 MangaOCR）─────────

def _get_annotation_font(size=40):
    """找一个能渲染编号的字体"""
    for fp in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _make_annotated_page(img_arr, coords, max_dim=1600):
    """在原图上画红色编号框，返回 (缩放后的 PNG bytes, scale)"""
    pil = Image.fromarray(img_arr).convert("RGB").copy()
    draw = ImageDraw.Draw(pil)
    font_size = max(24, int(min(pil.size) * 0.025))
    font = _get_annotation_font(font_size)

    for i, c in enumerate(coords):
        x1, y1, x2, y2 = [int(v) for v in c]
        num = str(i + 1)
        # 红色框
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        # 编号标签：放在左上角外侧（如果溢出顶边就放内侧）
        try:
            bbox = draw.textbbox((0, 0), num, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = font_size, font_size
        lx1 = x1 - 4
        ly1 = y1 - th - 8
        if ly1 < 2:
            ly1 = y1 + 2
        lx2 = lx1 + tw + 8
        ly2 = ly1 + th + 6
        draw.rectangle([lx1, ly1, lx2, ly2], fill=(255, 0, 0))
        draw.text((lx1 + 4, ly1 + 2), num, fill=(255, 255, 255), font=font)

    # 缩小到合理尺寸
    w, h = pil.size
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        pil = pil.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue(), scale


def _crop_bubble(img_arr, coord, pad_ratio=0.12, min_pad=15):
    x1, y1, x2, y2 = [int(v) for v in coord]
    ih, iw = img_arr.shape[:2]
    pad_x = max(min_pad, int((x2 - x1) * pad_ratio))
    pad_y = max(min_pad, int((y2 - y1) * pad_ratio))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(iw, x2 + pad_x)
    cy2 = min(ih, y2 + pad_y)
    return img_arr[cy1:cy2, cx1:cx2]


def translate_bubbles_via_vision(img_arr, coords, max_retries=2):
    """双输入 Vision 翻译：
    1) 整页带编号标注（提供全局场景上下文）
    2) 每个气泡的放大 crop（提供文字细节）

    这样 Gemini 既能看清数字/温度/卡牌等场景含义，也能读清笔画细节。
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=GEMINI_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    if not coords:
        return []

    # 1) 整页标注图
    page_bytes, _ = _make_annotated_page(img_arr, coords)
    page_b64 = base64.b64encode(page_bytes).decode()

    # 2) 每个气泡的 crop
    crops_b64 = []
    for c in coords:
        crop = _crop_bubble(img_arr, c)
        pil = Image.fromarray(crop)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        crops_b64.append(base64.b64encode(buf.getvalue()).decode())

    prompt_text = (
        f"这是一张日文漫画页。\n"
        f"第一张图是**整页缩略图**，我用红色数字 1 到 {len(coords)} 标注了每个需要翻译的气泡/文字区域。\n"
        f"后面跟着 {len(coords)} 张**放大的气泡截图**，按编号顺序排列。\n\n"
        f"请结合**整页场景**和**每张放大图**，为每个编号给出准确的中文翻译。\n\n"
        f"**特别重要**：\n"
        f"- **每个编号对应一个独立的气泡/文字区域**，互不混合。即使两个气泡在画面上靠得很近，\n"
        f"  也要严格只翻译当前编号方框内的内容，不要把旁边其他编号的文字带进来。\n"
        f"- **角色分类标签词**（如「長女」「次女」「三女」「お姉ちゃん」等单独出现的标签）\n"
        f"  对应的编号请只输出该词的中文（如「长女」「次女」「三女」），不要把它附加到\n"
        f"  其他气泡的翻译里。\n"
        f"- 数字、温度、时间、度数要先看整页场景（例如温度计显示、扑克牌点数、倒计时等），\n"
        f"  再确认字面内容，不要猜。例如气泡里的「37.3度」不能翻成「3度3分」。\n"
        f"- **数字中的小数点必须用 ASCII 点（.），不要用中文句号（。）**。\n"
        f"  正确：38.3度 / 错误：38。3度。\n"
        f"- 拟声词/叹词翻译成中文习惯的（うわあ→呜哇，きゃあ→呀，えっ→咦）。\n"
        f"- 不要保留日文长音符 ー，不要保留任何日文标点。\n"
        f"- 不要添加注释、括号说明、原文标注。\n"
        f"- 实在看不清的气泡返回空字符串 \"\"。\n"
        f"- 对人物设定/简介类的长段描述，完整翻译为一段连贯的中文，不要分段。\n\n"
        f"**输出格式**：严格 JSON 对象，键是编号字符串，值是中文翻译：\n"
        f'{{"1": "中文1", "2": "中文2", ..., "{len(coords)}": "中文{len(coords)}"}}\n'
        f"必须包含全部 {len(coords)} 个编号。只输出 JSON，不要加任何其他内容。"
    )

    content = [
        {"type": "text", "text": prompt_text},
        {"type": "text", "text": "【整页缩略图（红色数字为气泡编号）】"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_b64}"}},
    ]
    for i, b64 in enumerate(crops_b64):
        content.append({"type": "text", "text": f"【气泡 {i+1} 的放大图】"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
                max_tokens=3000,
                timeout=240
            )
            raw = resp.choices[0].message.content.strip()
            # 提取 JSON 对象
            m = re.search(r'\{[\s\S]*\}', raw)
            if m:
                result = json.loads(m.group())
                if isinstance(result, dict):
                    # 按编号顺序取回
                    out = []
                    for i in range(len(coords)):
                        v = result.get(str(i + 1), result.get(i + 1, ""))
                        out.append(str(v) if v is not None else "")
                    return out
        except Exception as e:
            last_error = e
            print(f"     ⚠ Gemini Vision 翻译尝试 {attempt+1}/{max_retries} 失败: {e}")
            continue

    print(f"     ✗ Vision 翻译全部失败，回退空字符串。最后错误: {last_error}")
    return [""] * len(coords)


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

    # Step 1: CTD 检测
    print("[1/6] Detecting...")
    det = requests.post(f"{API_BASE}/parallel/detect",
        json={"image": img_b64, "detector_type": "ctd"}, timeout=120).json()
    if not det.get("success"):
        print(f"  FAILED: {det.get('error')}"); return
    ctd_coords = list(det["bubble_coords"])
    ctd_polygons = list(det.get("bubble_polygons", []))
    ctd_directions = list(det.get("auto_directions", []))
    raw_mask = det.get("raw_mask")
    print(f"  CTD detected: {len(ctd_coords)} regions")

    # Step 2: OCR CTD 的结果（用于过滤 SFX / 非日文）
    print("[2/6] OCR CTD regions...")
    if ctd_coords:
        ocr = requests.post(f"{API_BASE}/parallel/ocr",
            json={"image": img_b64, "bubble_coords": ctd_coords, "source_language": SOURCE_LANG},
            timeout=120).json()
        if not ocr.get("success"):
            print(f"  FAILED: {ocr.get('error')}"); return
        ctd_texts = ocr.get("original_texts", [])
    else:
        ctd_texts = []
    while len(ctd_texts) < len(ctd_coords):
        ctd_texts.append("")

    # Vision 查漏补缺（基于已过滤的 CTD 结果，避免重复报告 SFX）
    ctd_valid_coords = [ctd_coords[i] for i in range(len(ctd_coords))
                        if is_bubble_text(ctd_coords[i], ctd_texts[i])]
    print("  Vision supplementary detection...")
    missed = detect_missed_bubbles_via_vision(img_pil, ctd_valid_coords)

    # 合并：CTD 原始 + Vision 新加
    all_coords = list(ctd_coords)
    all_polygons = list(ctd_polygons)
    all_directions = list(ctd_directions)
    all_texts = list(ctd_texts)

    # 先去重
    new_missed = [mc for mc in missed
                  if not any(bbox_iou(mc, c) > 0.25 for c in all_coords)]
    # 给新加的区域跑一次 OCR，方便后续过滤标签词
    if new_missed:
        try:
            ocr2 = requests.post(f"{API_BASE}/parallel/ocr",
                json={"image": img_b64, "bubble_coords": new_missed, "source_language": SOURCE_LANG},
                timeout=120).json()
            new_texts = ocr2.get("original_texts", []) if ocr2.get("success") else []
        except Exception:
            new_texts = []
    else:
        new_texts = []
    while len(new_texts) < len(new_missed):
        new_texts.append("__vision__")

    added_count = 0
    for mc, mt in zip(new_missed, new_texts):
        # 标签词在过滤阶段会被丢掉；非标签的统一给 __vision__ 占位（保证不被 SFX 过滤）
        cleaned = mt.strip().strip("「」『』（）()[]【】").strip()
        if cleaned in ROLE_LABEL_WORDS:
            print(f"  ⊘ Vision 区域是标签词，跳过: {mt}")
            continue
        all_coords.append(mc)
        all_polygons.append([])
        all_directions.append("v")
        all_texts.append("__vision__")
        added_count += 1
    print(f"  Vision added: {added_count} new regions (total: {len(all_coords)})")

    if not all_coords:
        print("  No text found."); return

    # Step 3: 颜色采样 + 过滤
    print("[3/6] Color & filter...")
    coords, polygons, directions, texts, text_colors, fg_rgbs = [], [], [], [], [], []
    for i in range(len(all_coords)):
        t = all_texts[i] if i < len(all_texts) else ""
        c = all_coords[i]
        w, h = c[2]-c[0], c[3]-c[1]
        # Vision 补检的区域绕过过滤（既然 Vision 看到了就信它）
        if t != "__vision__" and not is_bubble_text(c, t):
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

    # 渲染坐标 = 原始 coord（不扩展）
    # 之前用 expand_narrow_coords 会挤出气泡边界挡到画面。
    # 对横扁的标题/艺术字（宽 > 高 × 1.3），扩大高度让字号能撑起来。
    render_coords = []
    for i, c in enumerate(coords):
        x1, y1, x2, y2 = c
        bw, bh = x2 - x1, y2 - y1
        if bw > bh * 1.3 and bh < 150:
            # 横扁标题：高度扩展到原来的 2 倍（向两侧均匀），让 auto font size 给更大字号
            extra = int(bh * 1.0)
            y1 = max(0, y1 - extra // 2)
            y2 = min(img_h, y2 + extra // 2)
        render_coords.append([x1, y1, x2, y2])

    # Step 4: Gemini Vision 翻译（直接从气泡截图翻译，绕过 MangaOCR）
    print(f"[4/6] Vision translate via {MODEL}...")
    translated = translate_bubbles_via_vision(img_arr, coords)
    translated = [sanitize_translated_text(t) for t in translated]

    # 角色分类标签词不翻译也不擦除（保留原图），漫画读者能直接看懂
    SKIP_LABEL_WORDS = {
        "长女", "次女", "三女", "四女", "五女",
        "长姐", "次姐", "三姐", "二姐", "大姐",
        "大女儿", "二女儿", "三女儿",
        "長女", "長姉", "次姉",  # 日文汉字
    }
    for i, t in enumerate(translated):
        cleaned = t.strip().strip("「」『』（）()[]【】")
        if cleaned in SKIP_LABEL_WORDS:
            print(f"  ⊘ [{i+1}] 跳过角色标签: {t}")
            translated[i] = ""  # 清空 → 后续自动跳过擦除+渲染
    for i, (o, t) in enumerate(zip(texts, translated)):
        print(f"  [{i+1}] OCR={o[:40]}")
        print(f"       Vision={t}")

    # Step 5: 自适应背景色擦除（采样每个气泡的实际背景色）
    # 注意：用原始紧凑 coord 而不是扩展 coord，避免擦除/渲染超出气泡遮挡人物
    print("[5/6] Adaptive-bg cleaning...")
    inpaint_coords = list(coords)  # 不做扩展
    valid_mask = [bool(t and t.strip()) for t in translated]

    # 为每个气泡采样背景色（在原图的原始 coord 范围内取样更准）
    bg_colors = []
    for i in range(len(coords)):
        bg = sample_bubble_bg_color(img_arr, coords[i], polygons[i] if i < len(polygons) else None)
        bg_colors.append(bg)

    clean_pil = Image.fromarray(img_arr).copy()
    draw = ImageDraw.Draw(clean_pil)
    for i, c in enumerate(inpaint_coords):
        if not valid_mask[i]:
            continue
        fill_color = bg_colors[i]
        poly = polygons[i] if i < len(polygons) else None
        if poly and len(poly) >= 3:
            try:
                pts = [(int(p[0]), int(p[1])) for p in poly]
                draw.polygon(pts, fill=fill_color)
                continue
            except (TypeError, IndexError):
                pass
        # bbox 回退：圆角矩形 + 内缩 2px 避开气泡边线
        x1, y1, x2, y2 = c
        px1, py1 = max(0, int(x1)+2), max(0, int(y1)+2)
        px2, py2 = min(img_w, int(x2)-2), min(img_h, int(y2)-2)
        if px2 > px1 and py2 > py1:
            radius = max(6, int(min(px2-px1, py2-py1) * 0.18))
            try:
                draw.rounded_rectangle([px1, py1, px2, py2], radius=radius, fill=fill_color)
            except AttributeError:
                draw.rectangle([px1, py1, px2, py2], fill=fill_color)

    buf = io.BytesIO()
    clean_pil.save(buf, format="PNG")
    clean_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # Step 6: 渲染（先过滤空译文 —— Vision 补检可能产生空结果）
    print("[6/6] Rendering...")
    # 过滤掉空译文的气泡：这些可能是 Vision 误检或看不清的区域，
    # 跳过它们能避免白底擦除原图内容
    valid_indices = [i for i in range(len(coords))
                     if translated[i] and translated[i].strip()]
    skipped = len(coords) - len(valid_indices)
    if skipped > 0:
        print(f"  跳过 {skipped} 个空译文气泡")

    bubble_states = []
    final_coords = []
    final_polygons = []
    for i in valid_indices:
        # 文字方向：短文本一律横排，长文本按气泡形状决定
        rx1, ry1, rx2, ry2 = render_coords[i]
        bw, bh = rx2 - rx1, ry2 - ry1
        # 原始（未扩展）框尺寸 —— 用于判断是否是横扁标题
        ox1, oy1, ox2, oy2 = coords[i]
        obw, obh = ox2 - ox1, oy2 - oy1
        is_title = obw > obh * 1.3 and obh < 150
        tt = translated[i]
        tlen = len(tt)
        # 漫画中文约定：默认竖排。只有明确应该横排的情况才横排。
        has_number = bool(re.search(r'\d', tt))
        if has_number:
            td = "horizontal"
        elif tlen == 1:
            td = "horizontal"
        elif is_title:
            td = "horizontal"  # 横扁标题强制横排
        elif bw > bh * 1.5:
            td = "horizontal"
        else:
            td = "vertical"
        # 横扁标题用更大的字号
        font_size = 48 if is_title else 28
        final_coords.append(render_coords[i])
        final_polygons.append(polygons[i])
        bg_hex = "#{:02x}{:02x}{:02x}".format(*bg_colors[i])
        bubble_states.append({
            "originalText": texts[i],
            "translatedText": translated[i] if i < len(translated) else "",
            "textboxText": "",
            "coords": render_coords[i],
            "polygon": polygons[i],
            "fontSize": font_size,
            "fontFamily": FONT,
            "textDirection": td,
            "autoTextDirection": td,
            "textColor": text_colors[i],
            "fillColor": bg_hex,
            "rotationAngle": 0,
            "position": {"x": 0, "y": 0},
            "strokeEnabled": False,
            "strokeColor": "#FFFFFF",
            "strokeWidth": 0,
            "inpaintMethod": "solid",
            "autoFgColor": list(fg_rgbs[i]) if fg_rgbs[i] else None,
            "autoBgColor": None
        })

    render = requests.post(f"{API_BASE}/parallel/render",
        json={
            "clean_image": clean_b64,
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

    out_path = os.path.join(output_dir, os.path.splitext(name)[0] + "_translated.png")
    b64_to_file(render["final_image"], out_path)
    print(f"  Saved: {out_path}")


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

    # 断点续跑：跳过已有的输出
    done_count = 0
    failed = []
    for idx, fname in enumerate(images, 1):
        stem = os.path.splitext(fname)[0]
        out_ok = os.path.join(output_dir, stem + "_translated.png")
        out_review = os.path.join(output_dir, stem + "_translated_REVIEW.png")
        if os.path.exists(out_ok) or os.path.exists(out_review):
            print(f"\n[{idx}/{len(images)}] SKIP {fname} (已存在)")
            done_count += 1
            continue

        print(f"\n[{idx}/{len(images)}] {fname}")
        try:
            translate_image(os.path.join(input_dir, fname), output_dir)
            done_count += 1
        except Exception as e:
            print(f"  ✗ EXCEPTION: {type(e).__name__}: {e}")
            failed.append(fname)
            continue

    print(f"\n{'='*60}")
    print(f"All done! {done_count}/{len(images)} processed")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
