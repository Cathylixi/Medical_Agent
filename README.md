# 医疗报告OCR与结构化提取系统

基于大语言模型的医疗图像文字识别与结构化数据提取系统，支持超声心动图等医疗报告的智能分析。

## 🌟 项目特色

本系统实现了一个智能医疗报告处理流程，具有以下核心功能：

1. **📸 OCR文字识别** - 支持JPG图片和PDF文件的文字提取
2. **🤖 AI驱动的报告分类** - 自动识别超声心动图、冠脉CTA等报告类型
3. **📊 结构化数据提取** - 将非结构化文本转换为标准化表格数据
4. **🖥️ 可视化界面** - 提供GUI界面展示结构化结果
5. **🌍 灵活部署** - 支持国内外DashScope服务切换

## 🛠️ 环境要求

### 系统依赖
- Python 3.11+
- DashScope API密钥（阿里云通义千问）

### 推荐安装方式

#### 方法1：使用Conda环境（推荐）
```bash
# 1. 克隆项目
git clone https://github.com/Cathylixi/Medical_Agent.git
cd Medical_Agent

# 2. 创建conda环境
conda env create -f conda_environment
conda activate medical_agent
```

#### 方法2：使用pip安装
```bash
# 1. 克隆项目
git clone https://github.com/Cathylixi/Medical_Agent.git
cd Medical_Agent

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 配置API密钥

在项目根目录创建 `.env` 文件：
```bash
# DashScope API配置
DASHSCOPE_API_KEY=你的API密钥

# 站点配置（国内站）
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_OCR_MODEL=qwen-vl-ocr-latest
QWEN_TEXT_MODEL=qwen-max-0125

# 运行模式
DEBUG=0
AUTO_RUN=1
```

## 🚀 使用方法

### 批量处理PDF文件
```bash
PYTHONPATH=src python src/medical_agent/batch_pdf_import.py
```

### 批量处理JPG文件
```bash
PYTHONPATH=src python src/medical_agent/batch_jpg_import.py
```

### 单个处理JPG文件
```bash
PYTHONPATH=src python src/medical_agent/image_example.py
```

## ⚙️ 站点配置切换

本系统支持DashScope的国内站和国际站切换，只需修改 `.env` 文件：

### 🌐 国际站配置（海外用户）
```bash
DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
QWEN_OCR_MODEL=qwen-vl-ocr
QWEN_TEXT_MODEL=qwen-max-0125
```

### 🇨🇳 国内站配置（国内用户）
```bash
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_OCR_MODEL=qwen-vl-ocr-latest
QWEN_TEXT_MODEL=qwen-max-0125
```

## 📁 项目结构

```
Medical_Agent/
├── src/medical_agent/          # 核心代码
│   ├── agent.py               # 主要处理逻辑
│   ├── batch_jpg_import.py    # JPG批量处理
│   ├── batch_pdf_import.py    # PDF批量处理
│   ├── image_example.py       # 单文件处理示例
│   ├── gui.py                 # 图形界面
│   └── ...
├── data/                      # 数据文件
│   ├── test_jpg/             # 测试JPG文件
│   ├── 标准测量表.xlsx        # 标准测量项目配置
│   └── medical_terms.xlsx    # 医学术语知识库
├── requirements.txt           # Python依赖
├── conda_environment         # Conda环境配置
└── README.md                 # 项目说明
```

## 🎯 输出结果

系统会生成以下输出：

1. **结构化表格** - Excel格式的标准化数据
2. **OCR文本** - 提取的原始文字内容
3. **可视化界面** - GUI展示处理结果
4. **Parquet缓存** - 高效的数据存储格式

