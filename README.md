# Pixiv 漫画自动下载 + 翻译嵌字

> 上游 [Saber-Translator](https://github.com/MashiroSaber03/Saber-Translator) 的原 README 见 [`SABER_TRANSLATOR_README.md`](./SABER_TRANSLATOR_README.md)

基于 [Saber-Translator](https://github.com/MashiroSaber03/Saber-Translator) 后端的全自动日漫翻译流程：
**Pixiv 下载 → 检测气泡 → OCR → 翻译 → 修复擦字 → 中文嵌字 → 视觉审查**

| 功能 | 工具 |
|------|------|
| 一键下载 Pixiv 用户作品 | `download_pixiv.py` |
| 批量翻译嵌字 | `translate_manga.py` |
| 后端服务 | Saber-Translator (`app.py`) |

## 翻译效果

- **圆润可爱字体**：使用站酷快乐体 (ZCOOL KuaiLe)，贴近商业漫画翻译风格
- **保留原文颜色**：从像素采样文字颜色，区分不同角色对话色（女生红棕、男生深灰等）
- **智能过滤**：跳过音效字 (SFX)、时间标注 (AM 0:30)、非日文文字
- **窄框扩展**：检测框过窄时自动扩展，避免渲染文字过小
- **LAMA 修复**：使用 LAMA MPE / litelama 模型擦除原文
- **Gemini 翻译**：默认使用 `gemini-3.1-pro-preview`
- **质量审查闭环**：翻译后用 Gemini 视觉模型对比原图审查，不合格自动重试修复
- **多级重试 + 兜底**：3 级修复策略 + PIL 白底覆盖兜底

## 翻译管线

```
检测 (CTD) → OCR (MangaOCR) → 颜色采样 → 过滤 → 翻译 (Gemini)
                                                    ↓
渲染 (圆润字体+原色) ← 修复 (LAMA) ←──────────────┘
        ↓
   视觉审查 (Gemini)
        ↓
  ✓ 通过 → 保存
  ✗ 不合格 → 自动重试（3 级）→ 仍失败 → 白底兜底
```

## 重试策略详解

| 轮次 | mask_dilate | box_scale | 模型 | 说明 |
|------|------------|-----------|------|------|
| 1 | 10 | 1.0 | lama_mpe | 默认整框擦除 |
| 2 | 25 | 1.3 | lama_mpe | 框扩 1.3x + 加大擦除 |
| 3 | 50 | 1.6 | litelama | 框扩 1.6x + 换模型 + 最大擦除 |
| 兜底 | - | 1.15 | PIL | 在气泡区域画白色圆角矩形 |

## 智能过滤规则

| 规则 | 说明 | 示例 |
|------|------|------|
| 无日文内容 | OCR 结果不含假名/汉字 | "AM 6:00", "Amazon" |
| 极扁横条 | 宽高比 > 4 且高度 < 80px | 时间标注、页码 |
| 音效字 | 竖排宽度 < 75px 且文本 ≤ 4 字符 | "ああっ、", "ドキ" |
| 翻译失败重译 | 译文仍含日文假名 | 拟声词、感叹词 |

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

# LAMA 修复模型
mkdir -p models/lama
curl -L -o models/lama/inpainting_lama_mpe.ckpt \
  https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting_lama_mpe.ckpt
curl -L -o models/lama/big-lama.safetensors \
  https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.safetensors

# MangaOCR 模型会在首次运行时自动从 HuggingFace 下载
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

跟着提示在浏览器登录授权，拿到 refresh-token 即可。

之后下载某个用户的所有作品：

```bash
# 用数字 ID
python download_pixiv.py 19291125

# 用主页 URL
python download_pixiv.py https://www.pixiv.net/users/19291125

# 指定输出目录
python download_pixiv.py 19291125 my_downloads/
```

### 第三步：批量翻译

```bash
# 默认翻译 3sisters/ 目录到 3sisters/translated/
python translate_manga.py

# 指定输入/输出目录
python translate_manga.py pixiv_download/pixiv/<用户名>/ output/
```

翻译会经过完整管线：检测 → OCR → 颜色采样 → 过滤 → Gemini 翻译 → LAMA 修复 → 渲染 → Gemini 视觉审查（不合格自动重试）。

## 配置项 (.env)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GEMINI_API_KEY` | (必填) | Gemini API Key |
| `TRANSLATE_MODEL` | `gemini-3.1-pro-preview` | 翻译/审查模型 |
| `TRANSLATE_FONT` | `src/app/static/fonts/ZhanKuKuaiLeTi.ttf` | 渲染字体 |
| `TRANSLATE_TARGET_LANG` | `Chinese` | 目标语言 |
| `TRANSLATE_SOURCE_LANG` | `japanese` | 源语言 |
| `TRANSLATE_API_BASE` | `http://127.0.0.1:5000/api` | 后端地址 |

## 项目结构

```
manga_translator/
├── app.py                  # Saber-Translator 后端入口
├── download_pixiv.py       # Pixiv 下载脚本
├── translate_manga.py      # 翻译嵌字脚本（核心）
├── .env.example            # 配置模板
├── TRANSLATE_README.md     # 本文档
├── README.md               # 上游 Saber-Translator 文档
├── src/                    # Saber-Translator 后端源码
│   └── app/static/fonts/
│       └── ZhanKuKuaiLeTi.ttf  # 站酷快乐体（圆润字体）
├── models/                 # AI 模型目录（需手动下载）
│   ├── ctd/                # 文字检测
│   └── lama/               # 图像修复
├── pixiv_download/         # Pixiv 下载输出（gitignore）
└── 3sisters/               # 测试图片
    └── translated/         # 翻译结果（gitignore）
```

## 开发说明

### 翻译管线分步 API

`translate_manga.py` 调用 Saber-Translator 后端的并行 API：

| 步骤 | API | 作用 |
|------|-----|------|
| 检测 | `POST /api/parallel/detect` | CTD 检测气泡 |
| OCR | `POST /api/parallel/ocr` | MangaOCR 识别文字 |
| 翻译 | `POST /api/parallel/translate` | Gemini 批量翻译 |
| 单条重译 | `POST /api/translate_single_text` | SFX 专用 prompt 重译 |
| 修复 | `POST /api/parallel/inpaint` | LAMA 擦除原文 |
| 渲染 | `POST /api/parallel/render` | 渲染中文文字 |

### 客户端处理（不依赖后端）

- **像素颜色采样**：`extract_text_color()` - 直接读图像数组
- **过滤规则**：`is_bubble_text()` - 检测框尺寸 + OCR 内容
- **窄框扩展**：`expand_narrow_coords()` - 渲染前调整坐标
- **白底兜底**：`paint_white_patches()` - PIL 画圆角矩形
- **视觉审查**：`review_translated_image()` - 调 Gemini Vision API

## 致谢

- [Saber-Translator](https://github.com/MashiroSaber03/Saber-Translator) - 提供完整的漫画翻译后端框架
- [gallery-dl](https://github.com/mikf/gallery-dl) - Pixiv 下载支持
- [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator) - CTD / LAMA-MPE 模型
- [LAMA / litelama](https://github.com/advimman/lama) - 图像修复模型
- [MangaOCR](https://github.com/kha-white/manga-ocr) - 日文 OCR
- [站酷快乐体](https://www.zcool.com.cn/special/zcoolfonts/) - 免费商用圆润字体
- [Google Gemini](https://ai.google.dev/) - 翻译与视觉审查

## 注意事项

- Pixiv 下载需要登录授权，请遵守 Pixiv 使用条款
- 翻译结果仅供个人学习与研究，请勿用于商业用途
- LAMA 修复对 JPEG 压缩图片的复杂背景（如人物头发上的浮动文字）效果有限，已通过多级重试和白底兜底缓解
