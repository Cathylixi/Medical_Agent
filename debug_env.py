#!/usr/bin/env python3
"""
环境变量和配置调试脚本
用于快速检查系统配置状态
"""

import os
from pathlib import Path
from dotenv import load_dotenv

print("🔍 环境变量和配置调试检查")
print("=" * 50)

# 加载 .env 文件
print("📁 检查 .env 文件...")
env_path = Path(".env")
if env_path.exists():
    print(f"✅ .env 文件存在: {env_path.absolute()}")
    load_dotenv()
    with open(env_path, 'r') as f:
        env_content = f.read()
    print(f"📝 .env 文件内容预览:\n{env_content[:300]}...")
else:
    print("❌ .env 文件不存在")

print("\n🔧 关键环境变量检查:")
key_vars = [
    'DEBUG',
    'DASHSCOPE_API_KEY', 
    'DASHSCOPE_BASE_URL',
    'BAICHUAN_API_KEY',
    'BAICHUAN_BASE_URL'
]

for var in key_vars:
    value = os.getenv(var)
    if value:
        # 隐藏敏感信息，只显示前后几位
        if 'API_KEY' in var and len(value) > 10:
            masked_value = f"{value[:6]}...{value[-4:]}"
        else:
            masked_value = value
        print(f"  ✅ {var}={masked_value}")
    else:
        print(f"  ❌ {var}=未设置")

print("\n📊 文件系统检查:")
important_files = [
    "data/标准测量表.xlsx",
    "data/test_jpg",
    "src/medical_agent/agent.py",
    "src/medical_agent/table_format.py"
]

for file_path in important_files:
    path = Path(file_path)
    if path.exists():
        if path.is_file():
            print(f"  ✅ 文件存在: {file_path}")
        elif path.is_dir():
            jpg_count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.jpeg")))
            print(f"  ✅ 目录存在: {file_path} (含{jpg_count}个JPG文件)")
    else:
        print(f"  ❌ 不存在: {file_path}")

print("\n🧪 测试建议:")
print("1. 确保 DEBUG=0 以避免使用内置CTA测试文本")
print("2. 确保 data/test_jpg/ 目录包含真实的超声图片文件")
print("3. 检查 data/标准测量表.xlsx 是否包含超声相关测量项而非冠脉项")
print("4. 运行前执行: source .env (确保环境变量加载)")

print("\n🚀 推荐测试命令:")
print("  DEBUG=0 PYTHONPATH=src python src/medical_agent/batch_jpg_import.py")
