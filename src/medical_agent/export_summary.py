import os
from pathlib import Path
import pandas as pd

def show_export_summary():
    """
    显示双格式导出功能的概述
    """
    print("📊 医疗报告批量处理 - 双格式导出概述")
    print("=" * 60)
    
    # 检查parquet文件
    parquet_dir = Path("src/medical_agent/cache")
    parquet_files = list(parquet_dir.glob("*.parquet"))
    
    # 检查xlsx文件
    xlsx_dir = Path("exports/test_export")
    xlsx_files = list(xlsx_dir.glob("*.xlsx")) if xlsx_dir.exists() else []
    
    print(f"📁 Parquet文件目录: {parquet_dir}")
    print(f"   找到 {len(parquet_files)} 个 .parquet 文件")
    for f in parquet_files[-5:]:  # 显示最近5个
        print(f"   - {f.name}")
    if len(parquet_files) > 5:
        print(f"   ... 还有 {len(parquet_files) - 5} 个文件")
    
    print(f"\n📁 Excel文件目录: {xlsx_dir}")
    print(f"   找到 {len(xlsx_files)} 个 .xlsx 文件")
    for f in xlsx_files:
        print(f"   - {f.name}")
    
    print(f"\n✨ 功能特性:")
    print("   ✅ JPG批量处理 - 双格式导出 (parquet + xlsx)")
    print("   ✅ PDF批量处理 - 双格式导出 (parquet + xlsx)")
    print("   ✅ 自动创建导出目录")
    print("   ✅ 每个病例独立文件")
    print("   ✅ 文件名保持一致")
    
    print(f"\n🔄 使用方法:")
    print("   # JPG批量处理:")
    print("   PYTHONPATH=src AUTO_RUN=1 python src/medical_agent/batch_jpg_import.py")
    print()
    print("   # PDF批量处理:")
    print("   PYTHONPATH=src AUTO_RUN=1 python src/medical_agent/batch_pdf_import.py")
    
    print(f"\n📋 输出格式:")
    print("   - Parquet: 高效的二进制格式，用于程序读取")
    print("   - Excel: 人类友好的格式，便于查看和分享")

if __name__ == "__main__":
    show_export_summary() 