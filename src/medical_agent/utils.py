import json
import os
import pandas as pd
import time
from pathlib import Path

# Define the base paths
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, 'cache')
OCR_RESULT_DIR = os.path.join(ROOT_DIR, "../../exports/OCR_result")

def save_df_to_cache(df: pd.DataFrame, filename: str):
    """
    Save a DataFrame as a Parquet file to the cache directory.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the Parquet file (without extension).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    full_path = os.path.join(CACHE_DIR, f"{filename}.parquet")
    df.to_parquet(full_path, index=False)
    print(f"✅ Saved to {full_path}")

def load_df_from_cache(filename: str) -> pd.DataFrame:
    """
    Load a DataFrame from a Parquet file in the cache directory.
    
    Args:
        filename (str): The name of the Parquet file (without extension).
        
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    full_path = os.path.join(CACHE_DIR, f"{filename}.parquet")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ Cache file not found: {full_path}")
    df = pd.read_parquet(full_path)
    print(f"✅ Loaded from {full_path}")
    return df


def call_qwen_vl_api(state):
    completion = state["qwen"].chat.completions.create(
        model="qwen-vl-max-latest",
        messages=state['messages']
    )

    text = completion.choices[0].message.content
    return text


def safe_json_load(text):
    try:
        return json.loads(text)
    except:
        try:
            if text.startswith("```json"):
                text = text.split("```json")[1]
            if text.startswith("```"):
                text = text.split("```")[1] 
            if text.endswith("```"):
                text = text.split("```")[0]
            return json.loads(text)
        except Exception as e:
            print(f"Load Json Dict Failed. Error: {e}")
            return None

def save_ocr_result(ocr_text: str, filename: str, file_type: str = "unknown"):
    """
    保存OCR提取的原始文本到exports/OCR_result/目录
    
    Args:
        ocr_text (str): OCR提取的文本内容
        filename (str): 文件名（不含扩展名）
        file_type (str): 文件类型（jpg、pdf、text、batch_jpg、batch_pdf）
    """
    try:
        # 确保OCR结果目录存在
        os.makedirs(OCR_RESULT_DIR, exist_ok=True)
        
        # 生成带时间戳的文件名
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 根据文件类型生成不同的前缀
        type_prefix = {
            "jpg": "single_jpg",
            "pdf": "single_pdf", 
            "text": "text_input",
            "batch_jpg": "batch_jpg",
            "batch_pdf": "batch_pdf"
        }
        prefix = type_prefix.get(file_type, "unknown")
        
        # 构建完整的文件路径
        ocr_filename = f"{prefix}_{filename}_{timestamp}.txt"
        ocr_filepath = os.path.join(OCR_RESULT_DIR, ocr_filename)
        
        # 写入OCR文本
        with open(ocr_filepath, 'w', encoding='utf-8') as f:
            # 添加文件信息头部
            f.write(f"=== OCR提取结果 ===\n")
            f.write(f"源文件类型: {file_type}\n")
            f.write(f"源文件名: {filename}\n") 
            f.write(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n\n")
            f.write(ocr_text)
            
        print(f"💾 OCR结果已保存: {ocr_filepath}")
        return ocr_filepath
        
    except Exception as e:
        print(f"❌ 保存OCR结果失败: {e}")
        return None

def start_timer():
    """
    开始计时
    
    Returns:
        float: 开始时间戳
    """
    return time.time()

def end_timer_and_print(start_time: float, file_name: str, file_type: str = "file"):
    """
    结束计时并打印处理时间
    
    Args:
        start_time (float): 开始时间戳
        file_name (str): 文件名
        file_type (str): 文件类型描述
    """
    end_time = time.time()
    total_time = end_time - start_time
    
    # 格式化时间显示
    if total_time < 60:
        time_str = f"{total_time:.2f}秒"
    else:
        minutes = int(total_time // 60)
        seconds = total_time % 60
        time_str = f"{minutes}分{seconds:.2f}秒"
    
    print(f"⏱️  总处理时间 ({file_type}): {time_str} - {file_name}")

