# Pixiv 漫画自动下载 + 翻译嵌字

> 上游 [Saber-Translator](https://github.com/MashiroSaber03/Saber-Translator) 的原 README 见 [`SABER_TRANSLATOR_README.md`](./SABER_TRANSLATOR_README.md)

基于 [Saber-Translator](https://github.com/MashiroSaber03/Saber-Translator) 后端的全自动日漫翻译流水线：
**Pixiv 下载 → 检测气泡 → Gemini Vision 翻译 → 自适应背景擦除 → 中文嵌字**

| 功能 | 工具 |
|------|------|
| 一键下载 Pixiv 用户作品 / series | `download_pixiv.py` |
| 批量翻译嵌字 | `translate_manga.py` |
| 后端服务 | Saber-Translator (`app.py`) |

## 核心特性

- **圆润可爱字体**：使用站酷快乐体 (ZCOOL KuaiLe)，贴近商业漫画翻译风格
- **保留原文颜色**：从像素采样文字颜色，女生红棕、男生深灰自动区分
- **双输入 Vision 翻译**：同时发送**带编号标注的整页缩略图** + **每个气泡的放大截图**给 Gemini，结合全局场景上下文翻译，避免 OCR 误读数字/温度/卡牌点数
- **Vision 全面检测补漏**：CTD 漏检后 Gemini Vision 二次扫描，找出所有标题、角色简介段落、小气泡
- **自适应背景色擦除**：采样气泡实际背景色填充（而非硬白底），彩色气泡自然融入画面
- **智能过滤**：跳过音效字 (SFX)、时间标注 (AM 0:30)、非日文文字
- **方向智能判定**：按气泡形状 + 译文长度决定竖/横排，含数字一律横排避免小数点错位
- **文字清理**：sanitize 修正日文长音符 ー、装饰符号 ♡♪★、数字间的全角句号

## 翻译管线（B+B 方案）

```
CTD 检测 ─┬─→ MangaOCR (仅用于过滤 SFX/非日文)
         └─→ Vision 补漏扫描（捕获 CTD 漏掉的标题/简介/小气泡）
                    │
                    ↓
              颜色采样 + 过滤
                    │
                    ↓
          [Gemini Vision 翻译]
     整页带编号缩略图 + N 张气泡放大图
          → 返回 JSON 翻译结果
                    │
                    ↓
        背景色采样 + 气泡 polygon 白底擦除
                    │
                    ↓
              渲染中文译文 → 保存
```

相比早期版本的改进：
- **不再依赖 LAMA 修复**：直接在气泡 polygon 内填充采样的背景色，避免了复杂背景下的残留问题
- **不再依赖 MangaOCR 做翻译**：Vision 看图识字更准，尤其是数字、温度、时间等
- **不再需要重试循环 + 审查机制**：单次跑通，没有 `_REVIEW.png`

## 智能过滤规则

| 规则 | 说明 | 示例 |
|------|------|------|
| 无日文内容 | OCR 结果不含假名/汉字 | "AM 6:00", "Amazon" |
| 极扁横条 | 宽高比 > 4 且高度 < 80px | 时间标注、页码 |
| 音效字 | 竖排宽度 < 45px 且文本 ≤ 2 字符 | "ドキ"（但保留"むー"等短对话） |
| Vision 补检 | 绕过上述过滤 | CTD 漏检的标题/简介 |
| 空译文 | 跳过擦除与渲染，保留原图 | Vision 看不清返回的空字符串 |

## 渲染方向判定

| 条件 | 方向 |
|------|------|
| 译文含数字 | **横排**（避免小数点错位） |
| 译文长度 ≤ 5 字 | **横排**（避免窄竖排一字一列） |
| 气泡宽 > 高 × 1.1 | **横排** |
| 其他 | **竖排** |

## 环境要求

- Python 3.10+
- NVIDIA GPU（推荐，CPU 也可运行）
- Windows / Linux / macOS

## 安装

### 1. 创建虚拟环境

```bash
python -m venv venv
source venv/Scripts/activate  # Windows (Git Bash)
# venv\Scripts\activate.bat   # Windows (cmd)
# source venv/bin/activate    # Linux/Mac
```

### 2. 安装依赖

```bash
# GPU 版（推荐，需要 NVIDIA GPU）
pip install -r requirements-gpu.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# CPU 版
pip install -r requirements-cpu.txt

# 翻译脚本额外依赖
pip install python-dotenv
```

### 3. 下载所需模型

```bash
# CTD 文字检测模型
mkdir -p models/ctd
curl -L -o models/ctd/comictextdetector.pt \
  https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt

# MangaOCR 模型会在首次运行时自动从 HuggingFace 下载
# 注：当前管线已不依赖 LAMA 做擦除，无需下载 LAMA 模型
```

### 4. 构建前端（可选，仅 Web UI 需要）

```bash
cd vue-frontend
npm install
npm run build
cd ..
```

### 5. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 GEMINI_API_KEY
```

## 使用流程

### 第一步：启动 Saber-Translator 后端

```bash
# 终端 1
python app.py
```
后端会监听 `http://127.0.0.1:5000`，必须保持运行。

### 第二步：下载 Pixiv 作品

首次使用需要授权 Pixiv（一次性，refresh-token 会自动缓存）：

```bash
venv/Scripts/gallery-dl oauth:pixiv
```

跟着提示在浏览器登录授权即可。

支持多种下载粒度：

```bash
# 用数字 ID 下载用户全部作品
python download_pixiv.py 19291125

# 只下载某个 series（自动按 series 归类到子目录）
python download_pixiv.py "https://www.pixiv.net/user/19291125/series/292133"

# 单张作品
python download_pixiv.py "https://www.pixiv.net/artworks/143354025"

# 指定输出根目录
python download_pixiv.py 19291125 my_downloads/
```

目录结构：
```
pixiv_download/
└── pixiv/
    └── <用户ID> <用户名>/
        ├── series_<系列ID><系列标题>/   # series 作品归到子目录
        │   ├── 143354025_p0.jpg
        │   └── ...
        └── <作品ID>_p0.jpg              # 散图放用户根目录
```

### 第三步：批量翻译

```bash
# 默认翻译 3sisters/ 目录到 3sisters/translated/
python translate_manga.py

# 指定输入/输出目录
python translate_manga.py "pixiv_download/pixiv/19291125 takepoison5/series_292133妹" output/
```

支持**断点续跑**：已完成的图（输出目录里存在 `_translated.png`）会自动跳过，中途中断后直接重新运行即可。

## 配置项 (.env)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GEMINI_API_KEY` | (必填) | Gemini API Key |
| `TRANSLATE_MODEL` | `gemini-3.1-pro-preview` | Vision 翻译/检测模型 |
| `TRANSLATE_FONT` | `src/app/static/fonts/ZhanKuKuaiLeTi.ttf` | 渲染字体 |
| `TRANSLATE_TARGET_LANG` | `Chinese` | 目标语言 |
| `TRANSLATE_SOURCE_LANG` | `japanese` | 源语言 |
| `TRANSLATE_API_BASE` | `http://127.0.0.1:5000/api` | 后端地址 |

## 项目结构

```
manga_translator/
├── app.py                   # Saber-Translator 后端入口
├── download_pixiv.py        # Pixiv 下载脚本（支持 series/用户/单张）
├── translate_manga.py       # 翻译嵌字脚本（核心）
├── .env.example             # 配置模板
├── README.md                # 本文档
├── SABER_TRANSLATOR_README.md  # 上游 Saber-Translator 原文档
├── src/                     # Saber-Translator 后端源码
│   └── app/static/fonts/
│       └── ZhanKuKuaiLeTi.ttf  # 站酷快乐体
├── models/                  # AI 模型目录（需手动下载）
│   └── ctd/                 # CTD 文字检测
├── pixiv_download/          # Pixiv 下载输出（gitignore）
└── 3sisters/                # 测试图片
    └── translated/          # 翻译结果（gitignore）
```

## 开发说明

### 翻译管线分步 API

`translate_manga.py` 调用 Saber-Translator 后端的以下 API：

| 步骤 | API | 作用 |
|------|-----|------|
| 检测 | `POST /api/parallel/detect` | CTD 检测气泡坐标 + polygon |
| OCR | `POST /api/parallel/ocr` | MangaOCR 识别文字（仅用于过滤） |
| 渲染 | `POST /api/parallel/render` | 在擦除后的图上渲染中文 |

### 客户端处理（不依赖后端）

- **Vision 翻译**：`translate_bubbles_via_vision()` - 整页标注 + crop 双输入调 Gemini
- **Vision 补检**：`detect_missed_bubbles_via_vision()` - 用 Gemini 找 CTD 漏掉的区域
- **像素颜色采样**：`extract_text_color()` - 从暗像素取中位色作为文字颜色
- **背景色采样**：`sample_bubble_bg_color()` - 取气泡内亮像素中位色作为填充色
- **过滤规则**：`is_bubble_text()` - 根据尺寸 + OCR 内容过滤 SFX
- **窄框扩展**：`expand_narrow_coords()` - 竖排窄框横扩到 140px
- **文字清理**：`sanitize_translated_text()` - 修正日文标点/数字句号

## 已知限制

- **极密集小气泡**：多个小气泡堆在一起时，渲染器无法完美排版，字号偏小
- **艺术字/粗体装饰**：画面上的大号装饰字（如 `スマホ禁止！`）Vision 有时会跳过不翻
- **无对话页**：纯画面页（无任何文字）会跳过，不产生输出文件（正常行为）
- **JPEG 压缩图**：高度压缩的图片在背景采样时可能轻微失真

## 致谢

- [Saber-Translator](https://github.com/MashiroSaber03/Saber-Translator) - 提供完整的漫画翻译后端框架
- [gallery-dl](https://github.com/mikf/gallery-dl) - Pixiv 下载支持
- [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator) - CTD 检测模型
- [MangaOCR](https://github.com/kha-white/manga-ocr) - 日文 OCR
- [站酷快乐体](https://www.zcool.com.cn/special/zcoolfonts/) - 免费商用圆润字体
- [Google Gemini](https://ai.google.dev/) - 视觉翻译与场景理解

## 注意事项

- Pixiv 下载需要登录授权，请遵守 Pixiv 使用条款
- 翻译结果仅供个人学习与研究，请勿用于商业用途
