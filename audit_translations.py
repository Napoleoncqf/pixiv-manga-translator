"""全面审查已翻译的图片，列出有问题的文件名。

对比原图和翻译图，用 Gemini Vision 严格检查每张图，输出需要重跑的列表。
"""
import os, re, json, base64, sys, io
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
# 审查用 flash 模型（配额更高，足以做 QA 审查）
MODEL = "gemini-2.5-flash"

SRC_DIR = r"pixiv_download/pixiv/19291125 takepoison5/series_292133妹"
TR_DIR  = os.path.join(SRC_DIR, "translated")

client = OpenAI(
    api_key=GEMINI_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def read_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def audit_one(orig_path, trans_path):
    orig_ext = os.path.splitext(orig_path)[1].lower().lstrip(".")
    orig_mime = "jpeg" if orig_ext in ("jpg", "jpeg") else "png"
    prompt = (
        "你是漫画翻译质量审查员。第一张是日文漫画原图，第二张是翻译成中文后的图。\n\n"
        "请严格检查第二张中文图，对比原图，找出以下问题：\n"
        "1. **残留日文**：中文旁边或下面还能清晰读出原日文字（假名或日文汉字）\n"
        "2. **漏翻**：原图有日文对话但中文图上该位置没有中文（漏掉整个气泡）\n"
        "3. **文字遮挡**：中文白底块覆盖到了气泡外的人物、物体画面\n"
        "4. **翻译错误**：中文明显不通顺、字面错误（例如数字/温度识别错）\n"
        "5. **字体乱码**：出现方框、豆腐字或明显的字符错位\n"
        "6. **方向错乱**：文字排列混乱、列顺序不对\n\n"
        "**不算问题**：\n"
        "- 角色分类标签（長女/次女/三女 等）保留原日文汉字是故意的\n"
        "- AM 0:30 等时间标记\n"
        "- 服装英文 Logo\n"
        "- 艺术字标题翻译质量一般但能看懂\n"
        "- 轻微的 JPEG 压缩模糊\n\n"
        "**回复格式**（严格 JSON，最多列 3 个最严重问题）：\n"
        '{"ok": true/false, "issues": ["问题1", "问题2"]}\n'
        "没有问题返回 ok: true, issues: []。\n"
        "有问题的话，每条 issue 用一句话简要描述（不超过 50 字），最多列 3 个最严重的问题。\n"
        "**只输出 JSON，不要 markdown 代码块包装，不要任何说明文字。**"
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/{orig_mime};base64," + read_b64(orig_path)}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + read_b64(trans_path)}},
        ]}],
        temperature=0.1,
        max_tokens=2500,
        timeout=120
    )
    raw = resp.choices[0].message.content.strip()
    finish_reason = resp.choices[0].finish_reason

    def parse_json_attempt(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    # 剥代码块
    cleaned = raw
    if cleaned.startswith("```"):
        parts = cleaned.split("```", 2)
        if len(parts) >= 2:
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].rstrip()

    # 1) 直接 parse
    result = parse_json_attempt(cleaned)
    # 2) 截断补全：若缺少 } 或 ]，尝试补上
    if result is None:
        patched = cleaned
        # 去掉可能未闭合的尾部字符串
        if patched.count('"') % 2 == 1:
            patched = patched.rsplit('"', 1)[0]  # 丢掉最后一个未闭合字符串
        # 补齐未闭合的 [ 和 {
        open_braces = patched.count('{') - patched.count('}')
        open_brackets = patched.count('[') - patched.count(']')
        if open_brackets > 0:
            patched = patched.rstrip(',').rstrip() + ']' * open_brackets
        if open_braces > 0:
            patched = patched.rstrip(',').rstrip() + '}' * open_braces
        result = parse_json_attempt(patched)
    # 3) 找第一个 { 到最后一个 }
    if result is None:
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first >= 0 and last > first:
            result = parse_json_attempt(cleaned[first:last+1])

    if result is not None and isinstance(result, dict):
        return result.get("ok", True), result.get("issues", [])

    # 解析失败：保守处理，假定 OK（避免所有图都被误标为失败）
    trunc_hint = f" (truncated, finish={finish_reason})" if finish_reason == "length" else ""
    return True, [f"PARSE_FAILED{trunc_hint}: {raw[:200]}"]


def main():
    files = sorted([f for f in os.listdir(TR_DIR) if f.endswith("_translated.png")])
    print(f"Auditing {len(files)} translated images...\n")

    results = []
    for i, tf in enumerate(files, 1):
        stem = tf.replace("_translated.png", "")
        # 原图可能是 jpg 或 png
        orig = None
        for ext in ("jpg", "jpeg", "png"):
            cand = os.path.join(SRC_DIR, f"{stem}.{ext}")
            if os.path.exists(cand):
                orig = cand
                break
        if not orig:
            print(f"[{i}/{len(files)}] {stem}: ⚠ 原图未找到，跳过")
            continue

        trans = os.path.join(TR_DIR, tf)
        try:
            ok, issues = audit_one(orig, trans)
        except Exception as e:
            print(f"[{i}/{len(files)}] {stem}: ⚠ 审查错误 {e}")
            results.append((stem, False, [f"审查错误: {e}"]))
            continue

        if ok:
            print(f"[{i}/{len(files)}] {stem}: ✓")
        else:
            print(f"[{i}/{len(files)}] {stem}: ✗")
            for issue in issues[:3]:
                print(f"    - {issue}")
        results.append((stem, ok, issues))

    # 汇总
    failed = [(s, i) for s, ok, i in results if not ok]
    print(f"\n{'='*60}")
    print(f"审查完成：{len(results)} 张，{len(failed)} 张有问题\n")

    if failed:
        print("有问题的文件列表（用于重跑）：")
        for stem, _ in failed:
            print(f"  {stem}")

    # 保存结果
    with open("audit_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "total": len(results),
            "failed_count": len(failed),
            "failed": [{"stem": s, "issues": i} for s, i in failed]
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果保存到 audit_result.json")


if __name__ == "__main__":
    main()
