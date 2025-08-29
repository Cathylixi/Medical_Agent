import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from typing import List, Dict, Any
import glob
import pandas as pd
from agent import build_medical_agent, AgentState, init_llms, ocr_node, fill_form_node
from utils import save_df_to_cache, load_df_from_cache, save_ocr_result, start_timer, end_timer_and_print
from gui import show_popup_with_df
import json
import time

# Load environment variables
load_dotenv()

def process_single_jpg_to_parquet(image_path: str, output_name: str = None) -> bool:
    """
    处理单张 JPG 图片并保存结果到独立的 parquet 文件
    
    Args:
        image_path (str): 图片文件路径
        output_name (str): 输出文件名（不含扩展名），如果为None则使用图片文件名
        
    Returns:
        bool: 处理是否成功
    """
    # 开始计时
    start_time = start_timer()
    
    try:
        print(f"开始处理图片: {image_path}")
        
        # 检查文件是否存在
        img_file = Path(image_path)
        if not img_file.exists():
            print(f"❌ 图片文件不存在: {image_path}")
            return False
        
        # 确定输出文件名
        if output_name is None:
            output_name = img_file.stem  # 获取不含扩展名的文件名
        
        # 读取并编码图片
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        # 创建图片内容格式
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        
        # 创建初始状态
        initial_state = AgentState()
        initial_state['context'] = {
            'process_start_time': start_time,
            'current_file_name': img_file.name,
            'file_type': "单个JPG"
        }
        
        try:
            # 初始化智能体
            state = init_llms(initial_state)
            state['image_content'] = image_content
            
            # 设置OCR所需的messages格式
            state['messages'] = [
                {
                    "role": "user",
                    "content": [
                        image_content,
                        {"type": "text", "text": "Read all the text in the image"},
                    ],
                }
            ]
            
            # 运行OCR (不设置DEBUG模式，真实调用API)
            state = ocr_node(state)
            
            # 检查OCR结果
            if 'context' not in state or 'ocr' not in state['context']:
                print(f"⚠️ {image_path} OCR未获取到文本")
                end_timer_and_print(start_time, img_file.name, "单个JPG")
                return False
            
            # 保存OCR结果
            ocr_text = state['context']['ocr']
            save_ocr_result(ocr_text, output_name or img_file.stem, "jpg")
            
            print(f"✅ {img_file.name} OCR完成")
            
            # 运行结构化提取
            state = fill_form_node(state)
            
            # 手动保存结果到独立的parquet文件（使用自定义文件名）
            if 'formatted_table' in state:
                df = state['formatted_table']
                
                # 保存parquet文件
                save_df_to_cache(df, output_name)
                
                # 同时导出xlsx文件到exports目录
                export_dir = Path("exports/test_export")
                export_dir.mkdir(parents=True, exist_ok=True)
                xlsx_path = export_dir / f"{output_name}.xlsx"
                
                try:
                    df.to_excel(xlsx_path, index=False, engine='openpyxl')
                    print(f"✅ {img_file.name} 结构化完成，结果已保存到:")
                    print(f"   - Parquet: {output_name}.parquet")
                    print(f"   - Excel: {xlsx_path}")
                except Exception as e:
                    print(f"⚠️ Excel导出失败: {e}")
                    print(f"✅ {img_file.name} 结构化完成，结果已保存到 {output_name}.parquet")
                
                end_timer_and_print(start_time, img_file.name, "单个JPG")
                return True
            else:
                print(f"⚠️ {img_file.name} 结构化提取失败")
                end_timer_and_print(start_time, img_file.name, "单个JPG")
                return False
                
        except Exception as e:
            print(f"❌ 处理 {img_file.name} 时出错: {e}")
            end_timer_and_print(start_time, img_file.name, "单个JPG")
            return False
    
    except Exception as e:
        print(f"❌ 读取图片 {image_path} 时出错: {e}")
        end_timer_and_print(start_time, Path(image_path).name, "单个JPG")
        return False

def batch_process_jpg_directory(input_dir: str, output_dir: str = None) -> Dict[str, Any]:
    """
    批量处理目录下的所有 JPG 图片
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径，如果为None则使用默认缓存目录
        
    Returns:
        Dict[str, Any]: 处理结果统计
    """
    print("📁 JPG 批量处理工具")
    print("-" * 50)
    
    # 检查输入目录
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"❌ 输入目录不存在或不是目录: {input_dir}")
        return {"success": False, "error": "输入目录无效"}
    
    # 查找所有 JPG 文件
    jpg_patterns = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    jpg_files = []
    for pattern in jpg_patterns:
        jpg_files.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    if not jpg_files:
        print(f"⚠️ 在目录 {input_dir} 中未找到 JPG 文件")
        return {"success": False, "error": "未找到JPG文件"}
    
    print(f"🔍 找到 {len(jpg_files)} 个 JPG 文件:")
    for jpg_file in jpg_files:
        print(f"   - {Path(jpg_file).name}")
    
    # 处理结果统计
    results = {
        "total_files": len(jpg_files),
        "success_files": [],
        "failed_files": [],
        "success_count": 0,
        "failed_count": 0
    }
    
    # 批量处理每个文件
    for i, jpg_file in enumerate(jpg_files, 1):
        file_name = Path(jpg_file).name
        print(f"\n📝 [{i}/{len(jpg_files)}] 处理: {file_name}")
        
        # 生成输出文件名（避免重名）
        output_name = f"patient_{Path(jpg_file).stem}"
        
        # 处理单个文件（时间统计已在函数内部处理）
        success = process_single_jpg_to_parquet(jpg_file, output_name)
        
        if success:
            results["success_files"].append(file_name)
            results["success_count"] += 1
        else:
            results["failed_files"].append(file_name)
            results["failed_count"] += 1
        
        # 短暂延迟，避免API调用过快
        time.sleep(1)
    
    # 输出处理结果
    print("\n" + "=" * 50)
    print("📊 批量处理结果:")
    print(f"   总文件数: {results['total_files']}")
    print(f"   成功处理: {results['success_count']}")
    print(f"   处理失败: {results['failed_count']}")
    
    if results["success_files"]:
        print("\n✅ 成功处理的文件:")
        for file_name in results["success_files"]:
            print(f"   - {file_name}")
    
    if results["failed_files"]:
        print("\n❌ 处理失败的文件:")
        for file_name in results["failed_files"]:
            print(f"   - {file_name}")
    
    print(f"\n💾 结果文件保存在:")
    print(f"   - Parquet: src/medical_agent/cache/")
    print(f"   - Excel: exports/test_export/")
    print("   每个病例对应一个独立的文件")
    
    return results

def main():
    """
    主函数：批量处理 JPG 图片
    """
    print("🏥 医疗报告 JPG 批量处理工具")
    print("=" * 50)
    
    # 默认输入目录
    default_input_dir = "data/test_jpg"
    
    # 检查是否在自动模式
    if os.environ.get('AUTO_RUN', '0') == '1':
        input_dir = default_input_dir
        print(f"自动模式：使用默认目录 {input_dir}")
    else:
        # 获取用户输入
        input_dir = input(f"请输入JPG文件目录路径 (直接回车使用默认路径 '{default_input_dir}'): ").strip()
        if not input_dir:
            input_dir = default_input_dir
    
    # 开始批量处理
    results = batch_process_jpg_directory(input_dir)
    
    if results.get("success", True) and results["success_count"] > 0:
        print("\n🎉 批量处理完成！")
        print("💡 提示：你可以使用 GUI 工具查看每个病例的结构化结果")
    else:
        print("\n⚠️ 批量处理未成功完成，请检查错误信息")

if __name__ == "__main__":
    main() 