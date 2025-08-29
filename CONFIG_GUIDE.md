# 🌍 DashScope 配置指南

本项目支持灵活切换DashScope的国内站和国际站。只需修改 `.env` 文件即可完成切换。

## 📋 配置选项

### 🌐 国际站配置（默认）
适用于海外服务器或需要国际访问的环境：

```bash
# .env 文件配置
DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
QWEN_OCR_MODEL=qwen-vl-ocr
QWEN_TEXT_MODEL=qwen-max-0125
```

### 🇨🇳 国内站配置
适用于国内服务器环境：

```bash
# .env 文件配置
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_OCR_MODEL=qwen-vl-ocr-latest
QWEN_TEXT_MODEL=qwen-max-0148
```

## 🔄 快速切换方法

### 方法1：修改 .env 文件
1. 打开项目根目录下的 `.env` 文件
2. 找到以下三行配置：
   ```bash
   DASHSCOPE_BASE_URL=...
   QWEN_OCR_MODEL=...
   QWEN_TEXT_MODEL=...
   ```
3. 替换为对应站点的配置值
4. 保存文件并重新运行程序

### 方法2：使用 .env 文件中的注释模板
`.env` 文件中已预设好两套配置，只需：
1. 注释掉当前配置（在行首添加 `#`）
2. 取消注释目标配置（删除行首的 `#`）

## 🔑 API 密钥要求

**重要**：不同站点需要使用对应控制台生成的API密钥！

| 站点 | 控制台地址 | API密钥获取 |
|------|------------|-------------|
| 🌐 国际站 | https://dashscope-intl.aliyuncs.com | 国际站控制台 → API Key 管理 |
| 🇨🇳 国内站 | https://dashscope.aliyun.com | 国内站控制台 → API Key 管理 |

## ⚠️ 常见问题

### 401 错误：API密钥不正确
**原因**：使用了错误站点的API密钥
**解决**：确保API密钥来自正确的控制台

### 404 错误：模型不存在
**原因**：使用了错误站点的模型ID
**解决**：检查模型ID是否与站点匹配

### 网络超时
**原因**：网络环境与站点不匹配
**解决**：
- 国内网络 → 使用国内站配置
- 海外网络 → 使用国际站配置

## 🧪 测试配置

切换配置后，可以运行以下命令测试：

```bash
# 测试API连接
python fix_timeout.py

# 测试完整功能
PYTHONPATH=src python src/medical_agent/batch_jpg_import.py
```

## 📞 技术支持

如遇到配置问题，请检查：
1. ✅ API密钥是否来自正确的控制台
2. ✅ BASE_URL与API密钥站点是否匹配
3. ✅ 模型ID是否正确
4. ✅ 网络环境是否稳定
