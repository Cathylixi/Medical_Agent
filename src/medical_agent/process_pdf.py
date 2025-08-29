import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
import tempfile
from typing import List
from agent import build_medical_agent, AgentState
from utils import save_df_to_cache, load_df_from_cache, save_ocr_result, start_timer, end_timer_and_print
from gui import show_popup_with_df
import json
from medical_agent.utils import ROOT_DIR

# Load environment variables
load_dotenv()

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

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    图像预处理：灰度化、去噪、二值化，提高OCR识别准确率
    
    Args:
        image (Image.Image): 原始图片
        
    Returns:
        Image.Image: 处理后的图片
    """
    try:
        import cv2
        import numpy as np
        
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
        
        # 3. 可选：高斯模糊进一步去噪
        # denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
        
        # 4. 二值化 - 使用Otsu's方法自动选择阈值
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. 可选：形态学操作去除小噪点
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 将处理后的图像转回PIL Image格式
        processed_image = Image.fromarray(binary, mode='L')  # 'L'表示灰度模式
        
        # 转回RGB模式以保持与原有流程一致
        processed_image = processed_image.convert('RGB')
        
        return processed_image
        
    except ImportError:
        print("⚠️ OpenCV未安装，跳过图像预处理")
        # 如果OpenCV未安装，返回原图
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
        
    except Exception as e:
        print(f"⚠️ 图像预处理出错: {e}，使用原始图像")
        # 如果预处理出错，返回原图
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

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

def process_pdf_with_agent(pdf_path: str) -> None:
    """
    使用智能体处理PDF文件的完整流程
    
    Args:
        pdf_path (str): PDF文件路径
    """
    # 开始计时
    start_time = start_timer()
    
    print(f"开始处理PDF文件: {pdf_path}")
    
    # 检查文件是否存在
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ PDF文件不存在: {pdf_path}")
        end_timer_and_print(start_time, pdf_file.name, "单个PDF")
        return
    
    # 1. PDF转图片
    images = pdf_to_images(pdf_path)
    if not images:
        end_timer_and_print(start_time, pdf_file.name, "单个PDF")
        return
    
    # 2. 图像预处理
    processed_images = []
    for i, img in enumerate(images):
        processed_img = preprocess_image(img)
        processed_images.append(processed_img)
        print(f"✅ 预处理完成第{i+1}页")
    
    # 3. 初始化智能体
    medical_agent = build_medical_agent()
    
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
        
        # 设置DEBUG模式，跳过用户输入
        os.environ['DEBUG'] = '1'
        
        try:
            # 只运行到OCR节点，获取文本
            # 我们需要手动调用agent的各个节点
            from agent import init_llms_for_ocr, ocr_node
            
            # 初始化（OCR专用，不创建表格）
            state = init_llms_for_ocr(initial_state)
            state['image_content'] = image_content
            
            # 运行OCR
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
        
        finally:
            # 恢复DEBUG模式
            if 'DEBUG' in os.environ:
                del os.environ['DEBUG']
    
    # 4. 合并所有页面的OCR文本
    if all_ocr_texts:
        combined_text = "\n\n".join(all_ocr_texts)
        print(f"✅ 合并完成，共{len(all_ocr_texts)}页文本")
        
        # 保存OCR结果
        save_ocr_result(combined_text, pdf_file.stem, "pdf")
        
        # 5. 对合并后的文本进行结构化提取
        print("开始结构化提取...")
        
        # 创建一个新的状态用于结构化提取
        final_state = AgentState()
        from agent import init_llms, fill_form_node
        
        # 初始化
        final_state = init_llms(final_state)
        final_state['context'] = {'ocr': combined_text}
        
        try:
            # 运行结构化提取
            final_state = fill_form_node(final_state)
            print("✅ PDF处理完成，结果已保存到缓存并显示")
            end_timer_and_print(start_time, pdf_file.name, "单个PDF")
            
        except Exception as e:
            print(f"❌ 结构化提取时出错: {e}")
            end_timer_and_print(start_time, pdf_file.name, "单个PDF")
    else:
        print("❌ 没有获取到任何OCR文本")
        end_timer_and_print(start_time, pdf_file.name, "单个PDF")

def main():
    """
    主函数：获取用户输入的PDF路径并处理
    """
    print("PDF医疗报告处理工具")
    print("-" * 50)
    
    # 默认PDF路径
    default_pdf_path = os.path.join(ROOT_DIR, "../../data/智能分析用检查报告PDF文件(只有PDF文件)/报告例子/彩色超声报告单/20220824235959000_0000000001_YS100001_001.pdf")
    
    # 检查是否在自动模式（DEBUG环境变量）
    if os.environ.get('AUTO_RUN', '0') == '1':
        pdf_path = default_pdf_path
        print(f"自动模式：使用默认PDF文件 {pdf_path}")
    else:
        # 获取用户输入
        pdf_path = input(f"请输入PDF文件路径 (直接回车使用默认路径): ").strip()
        if not pdf_path:
            pdf_path = default_pdf_path
    
    # 检查文件扩展名
    if not pdf_path.lower().endswith('.pdf'):
        print("❌ 请输入有效的PDF文件路径")
        return
    
    # 处理PDF
    process_pdf_with_agent(pdf_path)

if __name__ == "__main__":
    main()
