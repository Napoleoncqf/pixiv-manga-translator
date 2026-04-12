"""
Pixiv 图片下载脚本

用法:
    python download_pixiv.py <user_id_or_url> [output_dir]

示例:
    python download_pixiv.py 19291125
    python download_pixiv.py https://www.pixiv.net/users/19291125
    python download_pixiv.py 19291125 my_pixiv/

依赖:
    - gallery-dl (已包含在 requirements 中)
    - 首次使用需先获取 refresh-token:
        venv/Scripts/gallery-dl oauth:pixiv
      浏览器登录授权后，token 会自动缓存。
"""
import os
import re
import sys
import argparse
import subprocess
from dotenv import load_dotenv

load_dotenv()


def normalize_user_url(arg: str) -> str:
    """把用户输入（数字 ID 或各种 URL 格式）规范化为 artworks URL。"""
    arg = arg.strip()
    if arg.isdigit():
        return f"https://www.pixiv.net/users/{arg}/artworks"
    m = re.search(r"users/(\d+)", arg)
    if m:
        return f"https://www.pixiv.net/users/{m.group(1)}/artworks"
    if arg.startswith("http"):
        return arg
    raise ValueError(f"无法解析: {arg}")


def download_user(user_arg: str, output_dir: str = "pixiv_download"):
    url = normalize_user_url(user_arg)
    os.makedirs(output_dir, exist_ok=True)

    # gallery-dl 路径
    gdl = os.path.join("venv", "Scripts", "gallery-dl")
    if not os.path.exists(gdl + ".exe") and not os.path.exists(gdl):
        gdl = "gallery-dl"  # fallback to PATH

    print(f"Pixiv 下载: {url}")
    print(f"保存到: {output_dir}/")
    print("=" * 60)

    cmd = [gdl, "-d", output_dir, url]
    try:
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"\n下载失败 (exit code {result.returncode})")
            print("如果是认证错误，请先运行: venv/Scripts/gallery-dl oauth:pixiv")
            sys.exit(result.returncode)
    except FileNotFoundError:
        print(f"找不到 gallery-dl: {gdl}")
        print("请先安装: pip install gallery-dl")
        sys.exit(1)

    # 统计结果
    total = 0
    for root, _, files in os.walk(output_dir):
        total += len([f for f in files if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".gif"))])
    print(f"\n{'='*60}")
    print(f"完成！共下载 {total} 个图片文件到 {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="下载 Pixiv 用户的所有插画作品",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("user", help="Pixiv 用户 ID 或主页 URL")
    parser.add_argument("output", nargs="?", default="pixiv_download",
                        help="输出目录 (默认: pixiv_download)")
    args = parser.parse_args()
    download_user(args.user, args.output)


if __name__ == "__main__":
    main()
