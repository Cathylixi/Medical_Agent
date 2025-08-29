import pandas as pd
import os
from pathlib import Path

def convert_parquet_to_xlsx():
    """
    批量转换cache文件夹下的所有parquet文件为xlsx格式
    """
    # 设置路径
    cache_dir = Path("src/medical_agent/cache")
    export_dir = Path("exports/parquet_to_xlsx")
    
    # 创建导出目录
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有parquet文件
    parquet_files = list(cache_dir.glob("*.parquet"))
    
    if not parquet_files:
        print("❌ 未找到任何parquet文件")
        return
    
    print(f"📁 找到 {len(parquet_files)} 个parquet文件")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    
    # 逐个转换
    for parquet_file in parquet_files:
        try:
            # 读取parquet文件
            df = pd.read_parquet(parquet_file)
            
            # 生成xlsx文件名
            xlsx_filename = parquet_file.stem + ".xlsx"
            xlsx_path = export_dir / xlsx_filename
            
            # 转换并保存为xlsx
            df.to_excel(xlsx_path, index=False, engine='openpyxl')
            
            print(f"✅ {parquet_file.name} → {xlsx_filename}")
            print(f"   数据形状: {df.shape} (行x列)")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 转换失败: {parquet_file.name}")
            print(f"   错误: {e}")
            error_count += 1
    
    print("=" * 60)
    print(f"🎉 转换完成！")
    print(f"✅ 成功: {success_count} 个文件")
    print(f"❌ 失败: {error_count} 个文件")
    print(f"📂 输出目录: {export_dir.absolute()}")

if __name__ == "__main__":
    print("🔄 开始批量转换parquet文件为xlsx格式...")
    convert_parquet_to_xlsx() 