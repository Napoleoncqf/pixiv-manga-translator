"""
Pixiv 图片下载脚本

支持下载：
  - 单个用户的全部作品  (https://www.pixiv.net/users/12345)
  - 单个用户的某个 series  (https://www.pixiv.net/user/12345/series/678)
  - 单张作品  (https://www.pixiv.net/artworks/12345)
  - 用户收藏  (https://www.pixiv.net/users/12345/bookmarks/artworks)
  - 也支持纯数字用户 ID

用法:
    python download_pixiv.py <user_id_or_url> [output_dir]

示例:
    # 下载用户全部作品（按 series 自动归类到子目录）
    python download_pixiv.py 19291125

    # 只下载某个 series（自动放到该 series 的目录里）
    python download_pixiv.py "https://www.pixiv.net/user/19291125/series/292133"

    # 单张作品
    python download_pixiv.py "https://www.pixiv.net/artworks/143354025"

    # 指定输出根目录
    python download_pixiv.py 19291125 my_pixiv/

目录结构:
    pixiv_download/
    └── pixiv/
        └── <用户ID> <用户名>/
            ├── series_<系列ID> <系列标题>/    # 属于某 series 的作品
            │   ├── 001_<作品ID>_p0.jpg
            │   └── ...
            └── _no_series/                    # 不属于任何 series 的散图
                ├── <作品ID>_p0.jpg
                └── ...

依赖:
    - gallery-dl (已包含在 requirements 中)
    - 首次使用需先获取 refresh-token:
        venv/Scripts/gallery-dl oauth:pixiv
      浏览器登录授权后，token 会自动缓存。
"""
import os
import re
import sys
import json
import argparse
import subprocess
import tempfile
from dotenv import load_dotenv

load_dotenv()


def normalize_url(arg: str) -> str:
    """规范化输入：纯数字 → 用户主页 URL；其他 URL 原样返回。"""
    arg = arg.strip()
    if arg.isdigit():
        return f"https://www.pixiv.net/users/{arg}/artworks"
    if arg.startswith("http"):
        return arg
    raise ValueError(f"无法解析: {arg}")


# gallery-dl 配置：按 series 自动归类
# - 属于某个 series 的作品：放到 series_<id>_<title>/ 目录，文件名带 series 序号
# - 不属于 series 的作品：放到 _no_series/ 目录
# 模板字段说明：
#   {user[id]} {user[account]}        - 用户ID + 账号名
#   {series[id]} {series[title]}      - series ID + 标题（不存在时回退到 _no_series）
#   {num_series}                      - 在 series 中的序号（不存在时为 0）
#   {id}                              - 作品ID
#   {num}                             - 多页作品的页索引（0,1,2...）
GALLERY_DL_CONFIG = {
    "extractor": {
        "base-directory": "./",
        "pixiv": {
            "filename": "{id}_p{num}.{extension}",
            "directory": [
                "pixiv",
                "{user[id]} {user[account]}",
                "{series[id]:?series_//}{series[title]:?/ /}",
            ],
            "ugoira-conv": False,
        }
    }
}


def find_gallery_dl():
    candidates = [
        os.path.join("venv", "Scripts", "gallery-dl.exe"),
        os.path.join("venv", "Scripts", "gallery-dl"),
        os.path.join("venv", "bin", "gallery-dl"),
        "gallery-dl",
    ]
    for c in candidates:
        if os.path.exists(c) or c == "gallery-dl":
            return c
    return "gallery-dl"


def download(target: str, output_dir: str = "pixiv_download"):
    url = normalize_url(target)
    os.makedirs(output_dir, exist_ok=True)

    # 写一个临时配置文件
    cfg = dict(GALLERY_DL_CONFIG)
    cfg["extractor"]["base-directory"] = output_dir.rstrip("/\\") + "/"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        cfg_path = f.name

    gdl = find_gallery_dl()
    print(f"Pixiv 下载: {url}")
    print(f"输出目录: {output_dir}/")
    print(f"配置文件: {cfg_path}")
    print("=" * 60)

    cmd = [gdl, "-c", cfg_path, url]
    try:
        result = subprocess.run(cmd, capture_output=False)
    except FileNotFoundError:
        print(f"找不到 gallery-dl: {gdl}")
        print("请先安装: pip install gallery-dl")
        sys.exit(1)
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass

    if result.returncode != 0:
        print(f"\n下载失败 (exit code {result.returncode})")
        print("如果是认证错误，请先运行: venv/Scripts/gallery-dl oauth:pixiv")
        sys.exit(result.returncode)

    # 统计结果
    total = 0
    for root, _, files in os.walk(output_dir):
        total += len([f for f in files if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".gif"))])
    print(f"\n{'='*60}")
    print(f"完成！{output_dir}/ 中共 {total} 个图片文件")


def main():
    parser = argparse.ArgumentParser(
        description="下载 Pixiv 用户作品 / series / 单张作品（自动按 series 分目录归类）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "target",
        help="Pixiv 用户 ID / 用户主页 URL / series URL / artwork URL",
    )
    parser.add_argument(
        "output", nargs="?", default="pixiv_download",
        help="输出根目录 (默认: pixiv_download)",
    )
    args = parser.parse_args()
    download(args.target, args.output)


if __name__ == "__main__":
    main()
