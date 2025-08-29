import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from typing import List, Dict, Any
import glob
import pandas as pd
from pdf2image import convert_from_path
from agent import build_medical_agent, AgentState, init_llms, ocr_node, fill_form_node
from utils import save_df_to_cache, load_df_from_cache, save_ocr_result, start_timer, end_timer_and_print
from gui import show_popup_with_df
import json
import time
import cv2
import numpy as np

# Load environment variables
load_dotenv()

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    图像预处理：灰度化、去噪、二值化，提高OCR识别准确率
    
    Args:
        image (Image.Image): 原始图片
        
    Returns:
        Image.Image: 处理后的图片
    """
    try:
        # PIL Image 转 OpenCV 格式 (numpy array)
        img_array = np.array(image)
        
        # 如果是RGB图像，转换为BGR格式（OpenCV标准）
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif img_array.ndim == 3 and img_array.shape[2] == 4:
            # 如果是RGBA，先转RGB再转BGR
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        else:
            # 如果已经是灰度图，转为BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        # 1. 灰度化
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 2. 去噪 - 使用中值滤波去除椒盐噪声
        denoised = cv2.medianBlur(gray, 3)
        
        # 3. 二值化 - 使用Otsu's方法自动选择阈值
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. 形态学操作去除小噪点
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 将处理后的图像转回PIL Image格式
        processed_image = Image.fromarray(binary, mode='L')  # 'L'表示灰度模式
        
        # 转回RGB模式以保持与原有流程一致
        processed_image = processed_image.convert('RGB')
        
        return processed_image
        
    except Exception as e:
        print(f"⚠️ 图像预处理出错: {e}，使用原始图像")
        # 如果预处理出错，返回原图
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    将PDF文件转换为图片列表
    
    Args:
        pdf_path (str): PDF文件路径
        dpi (int): 转换分辨率，默认300
        
    Returns:
        List[Image.Image]: 图片列表
    """
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"✅ 成功将PDF转换为{len(images)}页图片")
        return images
    except Exception as e:
        print(f"❌ PDF转换失败: {e}")
        return []

def image_to_base64(image: Image.Image) -> str:
    """
    将PIL图片转换为base64字符串
    
    Args:
        image (Image.Image): PIL图片对象
        
    Returns:
        str: base64编码的图片字符串
    """
    import io
    
    # 将图片保存到内存中的BytesIO对象
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='JPEG', quality=95)
    img_buffer.seek(0)
    
    # 转换为base64
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    return img_base64

def process_single_pdf_to_parquet(pdf_path: str, output_name: str = None) -> bool:
    """
    处理单个 PDF 文件并保存结果到独立的 parquet 文件
    
    Args:
        pdf_path (str): PDF文件路径
        output_name (str): 输出文件名（不含扩展名），如果为None则使用PDF文件名
        
    Returns:
        bool: 处理是否成功
    """
    # 开始计时
    start_time = start_timer()
    
    try:
        print(f"开始处理PDF文件: {pdf_path}")
        
        # 检查文件是否存在
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            print(f"❌ PDF文件不存在: {pdf_path}")
            end_timer_and_print(start_time, pdf_file.name, "单个PDF")
            return False
            
        # 将开始时间和文件信息添加到context中
        initial_state = AgentState()
        initial_state['context'] = {
            'process_start_time': start_time,
            'current_file_name': pdf_file.name,
            'file_type': "单个PDF"
        }
        
        # 确定输出文件名
        if output_name is None:
            output_name = pdf_file.stem  # 获取不含扩展名的文件名
        
        # 1. PDF转图片
        images = pdf_to_images(pdf_path)
        if not images:
            end_timer_and_print(start_time, pdf_file.name, "单个PDF")
            return False
        
        # 2. 图像预处理
        processed_images = []
        for i, img in enumerate(images):
            processed_img = preprocess_image(img)
            processed_images.append(processed_img)
            print(f"✅ 预处理完成第{i+1}页")
        
        # 收集所有页面的OCR文本
        all_ocr_texts = []
        
        for i, image in enumerate(processed_images):
            print(f"正在处理第{i+1}页...")
            
            # 转换图片为base64
            base64_image = image_to_base64(image)
            
            # 创建图片内容格式
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            
            # 创建初始状态
            initial_state = AgentState()
            initial_state['image_content'] = image_content
            initial_state['context'] = {
                'process_start_time': start_time,
                'current_file_name': pdf_file.name,
                'file_type': "单个PDF"
            }
            
            try:
                # 只运行到OCR节点，获取文本
                # 我们需要手动调用agent的各个节点
                from agent import init_llms_for_ocr, ocr_node
                
                # 初始化（OCR专用，不创建表格）
                state = init_llms_for_ocr(initial_state)
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
                
                # 获取OCR结果
                if 'context' in state and 'ocr' in state['context']:
                    ocr_text = state['context']['ocr']
                    all_ocr_texts.append(f"=== 第{i+1}页 ===\n{ocr_text}")
                    print(f"✅ 第{i+1}页OCR完成")
                else:
                    print(f"⚠️ 第{i+1}页OCR未获取到文本")
                    
            except Exception as e:
                print(f"❌ 处理第{i+1}页时出错: {e}")
        
        # 4. 合并所有页面的OCR文本
        if all_ocr_texts:
            combined_text = "\n\n".join(all_ocr_texts)
            print(f"✅ 合并完成，共{len(all_ocr_texts)}页文本")
            
            # 保存OCR结果
            save_ocr_result(combined_text, output_name or pdf_file.stem, "pdf")
            
            # 5. 对合并后的文本进行结构化提取
            print("开始结构化提取...")
            
            # 创建一个新的状态用于结构化提取
            final_state = AgentState()
            from agent import init_llms, fill_form_node
            
            # 初始化
            final_state = init_llms(final_state)
            final_state['context'] = {
                'ocr': combined_text,
                'process_start_time': start_time,
                'current_file_name': pdf_file.name,
                'file_type': "单个PDF"
            }
            
            try:
                # 运行结构化提取
                final_state = fill_form_node(final_state)
                
                # 手动保存结果到独立的parquet文件（使用自定义文件名）
                if 'formatted_table' in final_state:
                    df = final_state['formatted_table']
                    
                    # 保存parquet文件
                    save_df_to_cache(df, output_name)
                    
                    # 同时导出xlsx文件到exports目录
                    export_dir = Path("exports/test_export")
                    export_dir.mkdir(parents=True, exist_ok=True)
                    xlsx_path = export_dir / f"{output_name}.xlsx"
                    
                    try:
                        df.to_excel(xlsx_path, index=False, engine='openpyxl')
                        print(f"✅ {pdf_file.name} 结构化完成，结果已保存到:")
                        print(f"   - Parquet: {output_name}.parquet")
                        print(f"   - Excel: {xlsx_path}")
                    except Exception as e:
                        print(f"⚠️ Excel导出失败: {e}")
                        print(f"✅ {pdf_file.name} 结构化完成，结果已保存到 {output_name}.parquet")
                    
                    end_timer_and_print(start_time, pdf_file.name, "单个PDF")
                    return True
                else:
                    print(f"⚠️ {pdf_file.name} 结构化提取失败")
                    end_timer_and_print(start_time, pdf_file.name, "单个PDF")
                    return False
                
            except Exception as e:
                print(f"❌ 结构化提取时出错: {e}")
                end_timer_and_print(start_time, pdf_file.name, "单个PDF")
                return False
        else:
            print("❌ 没有获取到任何OCR文本")
            end_timer_and_print(start_time, pdf_file.name, "单个PDF")
            return False
            
    except Exception as e:
        print(f"❌ 读取PDF {pdf_path} 时出错: {e}")
        end_timer_and_print(start_time, Path(pdf_path).name, "单个PDF")
        return False

def batch_process_pdf_directory(input_dir: str, output_dir: str = None) -> Dict[str, Any]:
    """
    批量处理目录下的所有 PDF 文件
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径，如果为None则使用默认缓存目录
        
    Returns:
        Dict[str, Any]: 处理结果统计
    """
    print("📁 PDF 批量处理工具")
    print("-" * 50)
    
    # 检查输入目录
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"❌ 输入目录不存在或不是目录: {input_dir}")
        return {"success": False, "error": "输入目录无效"}
    
    # 查找所有 PDF 文件
    pdf_patterns = ['*.pdf', '*.PDF']
    pdf_files = []
    for pattern in pdf_patterns:
        pdf_files.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    if not pdf_files:
        print(f"⚠️ 在目录 {input_dir} 中未找到 PDF 文件")
        return {"success": False, "error": "未找到PDF文件"}
    
    print(f"🔍 找到 {len(pdf_files)} 个 PDF 文件:")
    for pdf_file in pdf_files:
        print(f"   - {Path(pdf_file).name}")
    
    # 处理结果统计
    results = {
        "total_files": len(pdf_files),
        "success_files": [],
        "failed_files": [],
        "success_count": 0,
        "failed_count": 0
    }
    
    # 批量处理每个文件
    for i, pdf_file in enumerate(pdf_files, 1):
        file_name = Path(pdf_file).name
        print(f"\n📝 [{i}/{len(pdf_files)}] 处理: {file_name}")
        
        # 生成输出文件名（避免重名）
        output_name = f"patient_pdf_{Path(pdf_file).stem}"
        
        # 处理单个文件
        success = process_single_pdf_to_parquet(pdf_file, output_name)
        
        if success:
            results["success_files"].append(file_name)
            results["success_count"] += 1
        else:
            results["failed_files"].append(file_name)
            results["failed_count"] += 1
        
        # 短暂延迟，避免API调用过快
        time.sleep(2)  # PDF处理通常更复杂，增加延迟
    
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
    主函数：批量处理 PDF 文件
    """
    print("🏥 医疗报告 PDF 批量处理工具")
    print("=" * 50)
    
    # 默认输入目录
    default_input_dir = "data/智能分析用检查报告PDF文件(只有PDF文件)/报告例子/彩色超声报告单"
    
    # 检查是否在自动模式
    if os.environ.get('AUTO_RUN', '0') == '1':
        input_dir = default_input_dir
        print(f"自动模式：使用默认目录 {input_dir}")
    else:
        # 获取用户输入
        input_dir = input(f"请输入PDF文件目录路径 (直接回车使用默认路径): ").strip()
        if not input_dir:
            input_dir = default_input_dir
    
    # 开始批量处理
    results = batch_process_pdf_directory(input_dir)
    
    if results.get("success", True) and results["success_count"] > 0:
        print("\n🎉 批量处理完成！")
        print("💡 提示：你可以使用 GUI 工具查看每个病例的结构化结果")
    else:
        print("\n⚠️ 批量处理未成功完成，请检查错误信息")

if __name__ == "__main__":
    main() 