from typing import List, Dict, Any, TypedDict, Literal, Union
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
import base64
from pathlib import Path
from openai import OpenAI
import os
from medical_agent.utils import call_qwen_vl_api, safe_json_load, end_timer_and_print
from medical_agent.utils import *
from medical_agent.table_format import create_formatted_df, ROW_INDEX
from typing import TypedDict, get_type_hints, Any
from medical_agent.prompts import FILL_IN_FORM_PROMPT, FILLIN_PROMPT_2, FILLIN_PROMPT_3, FILLIN_PROMPT_4, FILLIN_PROMPT_5, REPORT_CLASSIFIER_PROMPT, ULTRASOUND_EXTRACT_PROMPT
import json
import pandas as pd
from medical_agent.gui import show_popup_with_df
from medical_agent.utils import ROOT_DIR
from rapidfuzz import fuzz, process

# Define message types
class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: Union[str, Dict[str, Any]]
    content_type: Literal["text", "image"] = "text"


# Define the system prompt
SYSTEM_PROMPT = """你是一个医疗助手，你的主要任务是帮助医生整理病例，诊断报告等文件或图片，并将其整理成结构化数据存储起来。必要时，你也可以回答用户的医疗问题。如果可能，在回答医疗问题时请尽量提供来源。"""

def process_ultrasound_location(location, ocr, qwen, row_index, system_prompt, model_name="qwen-max-0125"):
    """处理单个超声测量项目的函数，用于并行执行"""
    if location not in row_index:
        return None, None, None
        
    ridx = row_index[location]
    
    # 动态生成别名映射规则
    try:
        from config_loader import generate_alias_prompt_section
        dynamic_alias_rules = generate_alias_prompt_section()
    except (ImportError, Exception):
        # 不使用任何别名规则，让AI进行精确匹配
        dynamic_alias_rules = ""
    
    input_prompt = ULTRASOUND_EXTRACT_PROMPT.format(
        ocr_text=ocr, 
        location=location,
        dynamic_alias_rules=dynamic_alias_rules
    )
    
    # 实现重试机制（指数退避）
    max_retries = 3
    attempt = 0
    
    while attempt < max_retries:
        try:
            completion = qwen.chat.completions.create(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': input_prompt}
                ]
            )
            text = completion.choices[0].message.content
            tmp = safe_json_load(text)
            return location, ridx, tmp
        except Exception as e:
            attempt += 1
            if attempt < max_retries:
                # 指数退避：等待时间为 2^attempt 秒，最大等待32秒
                wait_time = min(2 ** attempt, 32)
                import time
                time.sleep(wait_time)
            else:
                print(f"❌ {location} 处理失败: {e}")
                return location, ridx, None
    
    return location, ridx, None

def calculate_ea_ratios(formatted_table):
    """
    自动计算二尖瓣/三尖瓣的E/A比值
    
    Args:
        formatted_table (pd.DataFrame): 格式化的表格
        
    Returns:
        pd.DataFrame: 添加了E/A比值计算的表格
    """
    # 定义E峰和A峰的搜索模式
    valve_patterns = {
        "二尖瓣": {
            "e_patterns": ["二尖瓣E", "MV E", "Mitral E", "二尖瓣E峰", "MV E峰"],
            "a_patterns": ["二尖瓣A", "MV A", "Mitral A", "二尖瓣A峰", "MV A峰"],
            "ratio_name": "二尖瓣E/A比值",
            "ratio_english": "MV E/A ratio",
            "ratio_abbr": "MV E/A"
        },
        "三尖瓣": {
            "e_patterns": ["三尖瓣E", "TV E", "Tricuspid E", "三尖瓣E峰", "TV E峰"],
            "a_patterns": ["三尖瓣A", "TV A", "Tricuspid A", "三尖瓣A峰", "TV A峰"],
            "ratio_name": "三尖瓣E/A比值",
            "ratio_english": "TV E/A ratio", 
            "ratio_abbr": "TV E/A"
        }
    }
    
    ea_ratios_added = []
    
    for valve_name, patterns in valve_patterns.items():
        # 查找E峰和A峰的数值
        e_value = None
        a_value = None
        e_row_idx = None
        a_row_idx = None
        
        # 搜索E峰
        for idx, row in formatted_table.iterrows():
            name = str(row['名称']).strip()
            value = str(row['数值']).strip()
            
            # 检查是否匹配E峰模式且有数值
            for pattern in patterns["e_patterns"]:
                if pattern in name and value and value != "" and value != "-" and value != "NO":
                    try:
                        e_value = float(value)
                        e_row_idx = idx
                        break
                    except ValueError:
                        continue
            if e_value is not None:
                break
        
        # 搜索A峰  
        for idx, row in formatted_table.iterrows():
            name = str(row['名称']).strip()
            value = str(row['数值']).strip()
            
            # 检查是否匹配A峰模式且有数值
            for pattern in patterns["a_patterns"]:
                if pattern in name and value and value != "" and value != "-" and value != "NO":
                    try:
                        a_value = float(value)
                        a_row_idx = idx
                        break
                    except ValueError:
                        continue
            if a_value is not None:
                break
        
        # 如果找到了完整的E峰和A峰数值，计算E/A比值
        if e_value is not None and a_value is not None and a_value != 0:
            ea_ratio = round(e_value / a_value, 2)
            
            # 检查是否已经存在对应的E/A比值行
            ratio_exists = False
            for idx, row in formatted_table.iterrows():
                name = str(row['名称']).strip()
                if patterns["ratio_name"] in name or patterns["ratio_abbr"] in name:
                    # 更新现有的比值
                    formatted_table.at[idx, '数值'] = str(ea_ratio)
                    ratio_exists = True
                    break
            
            # 如果不存在，添加新行
            if not ratio_exists:
                new_row = {
                    "名称": f"{patterns['ratio_name']}({patterns['ratio_abbr']})",
                    "英文": patterns["ratio_english"],
                    "斑块种类": "",
                    "类型": "",
                    "症状": "",
                    "数值": str(ea_ratio),
                    "单位": "",  # E/A比值无单位
                    "狭窄程度": "",
                    "闭塞": "否"
                }
                
                # 使用pd.concat添加新行到表格开头
                new_row_df = pd.DataFrame([new_row])
                formatted_table = pd.concat([new_row_df, formatted_table], ignore_index=True)
                
                ea_ratios_added.append(f"{patterns['ratio_name']}: {ea_ratio}")
    
    return formatted_table

def separate_value_and_unit(value_str):
    """
    将带单位的数值字符串分离成纯数值和单位
    
    Args:
        value_str (str): 带单位的数值字符串，如 "700m/s", "500x300cm", "18mmHg"
    
    Returns:
        tuple: (纯数值, 单位)
        
    Examples:
        "700m/s" -> ("700", "m/s")
        "500x300cm" -> ("500x300", "cm") 
        "18mmHg" -> ("18", "mmHg")
        "59%" -> ("59", "%")
        "18" -> ("18", "")
    """
    import re
    
    if not value_str or value_str == "-" or value_str == "NO":
        return value_str, ""
    
    value_str = str(value_str).strip()
    
    # 定义常见的医学单位（按长度从长到短排序，优先匹配较长的单位）
    common_units = [
        'mmHg', 'ml/m²', 'cm/s', 'mm/s', 'm/s', 'mmHg/s', 'msec', 
        'cm²', 'mm²', 'm²', 'bpm', 'cm', 'mm', 'm', 'ml', 'kPa', 
        'Hz', 'sec', 'min', '%', 'g', 'kg', 'l'
    ]
    
    # 先尝试找到最长匹配的单位后缀
    matched_unit = ""
    numeric_part = value_str
    
    for unit in common_units:
        # 检查字符串是否以这个单位结尾
        if value_str.lower().endswith(unit.lower()):
            # 提取数值部分（去掉单位后的部分）
            potential_numeric = value_str[:-len(unit)].strip()
            
            # 验证数值部分是否是有效的数值格式（包括复合格式如 200x160）
            # 允许：数字、小数点、x、<、>、空格
            if re.match(r'^[0-9]+(?:\.[0-9]+)?(?:\s*[x×]\s*[0-9]+(?:\.[0-9]+)?)*\s*[<>]?\s*$', potential_numeric, re.IGNORECASE):
                matched_unit = unit
                numeric_part = potential_numeric.strip()
                break
    
    # 如果没有找到标准单位，但字符串包含数字，尝试分离
    if not matched_unit:
        # 匹配模式：数字部分 + 非数字部分
        match = re.match(r'^([0-9]+(?:\.[0-9]+)?(?:\s*[x×]\s*[0-9]+(?:\.[0-9]+)?)*\s*[<>]?\s*)(.*?)$', value_str, re.IGNORECASE)
        if match:
            numeric_part = match.group(1).strip()
            unit_part = match.group(2).strip()
            if unit_part:  # 如果有单位部分
                matched_unit = unit_part
    
    return numeric_part, matched_unit

class AgentState(TypedDict):
    messages: list
    qwen: Any
    gpt: Any
    formatted_table: Any
    row_index: dict
    image_content: dict
    context: dict
    next: str

# Dynamically create an initial state with sensible defaults
def init_typed_dict(cls: TypedDict):
    hints = get_type_hints(cls)
    default_values = {
        list: [],
        dict: {},
        str: "",
        int: 0,
        float: 0.0,
        bool: False,
    }

    state = {}
    for key, hint in hints.items():
        # handle special case of typing.List, typing.Dict, etc.
        origin = getattr(hint, '__origin__', hint)
        state[key] = default_values.get(origin, None)
    return state


def init_llms(state: AgentState):
    """
    Initialize dual-client setup: OCR client + Medical LLM client
    """
    state = init_typed_dict(AgentState)

    # OCR 专用客户端 (继续使用 Qwen-VL)
    state["ocr_client"] = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url=os.getenv('DASHSCOPE_BASE_URL', "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    )
    
    # 医疗推理专用客户端 (使用本地部署的 Baichuan M2)
    # 如果本地服务不可用，会自动fallback到OCR客户端
    baichuan_base_url = os.getenv('BAICHUAN_BASE_URL', 'http://localhost:8000/v1')
    state["medical_llm"] = OpenAI(
        api_key=os.getenv('BAICHUAN_API_KEY', 'not-needed'),  # 本地部署通常不需要
        base_url=baichuan_base_url,
    )
    
    # 保留原有的 qwen 客户端作为备选 (向后兼容)
    state["qwen"] = state["ocr_client"]  # 指向OCR客户端，保持兼容性
    
    # 保留原有的 GPT 客户端
    state["gpt"] = ChatOpenAI(
        model="gpt-4o",
        max_tokens=1024,
        temperature=0.7
    )

    state['messages'].append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })

    state['formatted_table'] = create_formatted_df()
    from medical_agent.table_format import get_dynamic_row_index
    state['row_index'] = get_dynamic_row_index()

    return state

def init_llms_for_ocr(state: AgentState):
    """
    Initialize OCR-only client for pure OCR tasks
    """
    state = init_typed_dict(AgentState)

    # OCR 专用客户端 (继续使用 Qwen-VL)
    state["ocr_client"] = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url=os.getenv('DASHSCOPE_BASE_URL', "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    )
    
    # 保留原有的 qwen 客户端指向OCR客户端，保持兼容性
    state["qwen"] = state["ocr_client"]
    
    # 保留原有的 GPT 客户端
    state["gpt"] = ChatOpenAI(
        model="gpt-4o",
        max_tokens=1024,
        temperature=0.7
    )

    state['messages'].append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })

    # 注释掉OCR阶段的表格创建，只在最终分析时创建
    # state['formatted_table'] = create_formatted_df()
    # state['row_index'] = ROW_INDEX

    return state

# Define the response generation node
def create_input_node(state: AgentState):
    """Create a node for handling user input and generating responses."""
    
    # Default values
    default_image_path = os.path.join(ROOT_DIR, "../../data/input_2.jpg")
    default_question = "请分析这张医疗图像并提供诊断建议。"
    
    image_path = default_image_path
    question = default_question
    
    # Get user input or use defaults
    # DEBUG 模式下跳过输入
    if os.environ.get('DEBUG', '0') == '0':
        try:
            user_image_path = input("请输入图片路径 (直接回车使用默认路径): ").strip()
            user_question = input("请输入您的问题 (直接回车使用默认问题): ").strip()
            if user_image_path:
                image_path = user_image_path
            if user_question:
                question = user_question
        except:
            pass
    
    # Read and encode the image
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"警告：找不到图片 {image_path}，请重新输入")
            raise RuntimeError(f"找不到图片 {image_path}")
            
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Create image content in the format expected by the API
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }

        state['image_content'] = image_content
        
        # Add the message to state
        state["messages"].append({
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": question}
            ],
            "content_type": "image"
        })
        
    except Exception as e:
        print(f"处理图片时出错: {e}")
        # If image processing fails, fall back to text-only
        state["messages"].append({
            "role": "user",
            "content": question,
            "content_type": "text"
        })
    
    return state


def ocr_node(state: AgentState):
    """Create a node for OCR using dedicated OCR client."""
    image_content = state['image_content']
    if not image_content:
        print("Warning: There is no image content to process. Skipping OCR.")
        return state

    # Call the OCR API using dedicated OCR client
    messages=[
        {
            "role": "user",
            "content": [
                image_content,
                # 为保证识别效果，如果使用qwen-vl-ocr 系列模型， 目前模型内部会统一使用"Read all the text in the image."进行识别，用户输入的文本不会生效。
                {"type": "text", "text": "Read all the text in the image"},
            ],
        }
    ]

    print("正在调用OCR模型获取文本提取结果")
    
    # 🔍 调试信息：检查DEBUG环境变量
    debug_mode = os.environ.get('DEBUG', '0')
    print(f"🔍 DEBUG模式状态: DEBUG={debug_mode}")
    
    if debug_mode == '1':
        text = """CT检查报告单
(扫码查看图像
检查号：220901
申请科室：内分泌科Ⅱ病区
申请医生：张玲
姓名： 住院号：2022041478 年龄：55岁 性别：男 床号：5
检查项目：256排冠状动脉CTA
检查所见：
冠状动脉呈右优势型。左主干起源于左窦，右冠状动脉起源于右窦。
左主干管壁可见钙化斑块，管腔轻微狭窄约10%。左前降支近段管壁可见钙化斑块，管腔轻度狭窄约25%；中段管壁可见混合斑块，管腔重度狭窄约85%；远段管壁可见钙化斑块，管腔轻度狭窄约25%。第一，第二对角支未见斑块及明显狭窄。左回旋支中远段管壁可见非钙化斑块，管腔轻度狭窄约25%；近段未见斑块及明显狭窄。第一，第二钝缘支未见斑块及明显狭窄。中间支未见斑块及明显狭窄。
右冠状动脉近段管壁可见钙化、非钙化斑块，管腔轻度狭窄约25%；中段、远段未见斑块及明显狭窄。右后降支未见斑块及明显狭窄。左室后支未见斑块及明显狭窄。
心脏各腔室不大，心肌未见异常密度影。
印象：
冠状动脉CTA：1.左主干管壁钙化斑块，管腔轻微狭窄。2.左前降支近段管壁钙化斑块，管腔轻度狭窄；中段管壁混合斑块，管腔重度狭窄；远段管壁钙化斑块，管腔轻度狭窄。3.左回旋支中远段管壁非钙化斑块，管腔轻度狭窄。4.右冠状动脉近段管壁钙化、非钙化斑块，管腔轻度狭窄。
检查日期：2022-09-08 审核医师：
报告医师：
注：1.本报告仅供临床科室申请医生诊治参考！
2.二维码链接图像，请妥善保存本报告！
报告时间：2022-09-08 17:19:39
        """
    else:
        # 使用专门的 OCR 客户端
        completion = state["ocr_client"].chat.completions.create(
            model="qwen-vl-ocr",
            messages=messages
        )
        text = completion.choices[0].message.content
    
    # 🔍 调试信息：显示OCR提取结果的前200字符
    print(f"🔍 OCR提取文本预览: {text[:200]}...")
    print(f"🔍 OCR文本总长度: {len(text)} 字符")

    state['context']['ocr'] = text
    return state


def get_medical_llm_client(state: AgentState):
    """
    智能选择文本理解客户端
    暂时全部使用Qwen，保证稳定性
    """
    # 直接使用Qwen客户端，确保稳定性
    return state["qwen"], "qwen-max-0125"


def fill_form_node(state: AgentState):
    """
    智能分流版本的表格填充节点
    
    工作流程：
    1. 先识别报告类型（CTA 或 超声）
    2. 根据报告类型选择相应的处理逻辑
    3. 输出统一的格式
    """
    print("🚀 开始智能结构化提取...")
    
    # 获取基本数据
    ocr = state['context']['ocr']
    formatted_table = state['formatted_table']
    row_index = state['row_index']
    
    # 智能选择文本理解客户端
    medical_client, medical_model = get_medical_llm_client(state)
    qwen = medical_client  # 保持向后兼容
    
    # 🔍 调试信息：显示使用的模型
    print(f"🔍 使用的AI模型: {medical_model}")
    print(f"🔍 OCR文本中是否包含'冠状动脉': {'冠状动脉' in ocr}")
    print(f"🔍 OCR文本中是否包含'超声': {'超声' in ocr}")
    print(f"🔍 OCR文本中是否包含'心动图': {'心动图' in ocr}")
    
    # ==============================================================
    # 第一步：智能识别报告类型
    # ==============================================================
    print("📋 正在识别报告类型...")
    
    classifier_prompt = REPORT_CLASSIFIER_PROMPT.format(ocr_text=ocr)
    try:
        completion = qwen.chat.completions.create(
            model=medical_model,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': classifier_prompt}
            ]
        )
        classifier_text = completion.choices[0].message.content
        # print(f"分类结果原文: {classifier_text}")
        
        # 🔍 调试信息：显示分类器原始响应
        print(f"🔍 分类器原始响应: {classifier_text}")
        
        classification_result = safe_json_load(classifier_text)
        if classification_result and 'report_type' in classification_result:
            report_type = classification_result['report_type']
            confidence = classification_result.get('confidence', '未知')
            reason = classification_result.get('reason', '无理由')
            print(f"✅ 报告类型识别完成: {report_type} (置信度: {confidence})")
            print(f"   判断理由: {reason}")
        else:
            print("⚠️ 报告类型识别失败，默认按CTA处理")
            print(f"🔍 分类结果解析失败，原始内容: {classification_result}")
            report_type = "CTA"
    except Exception as e:
        print(f"❌ 报告类型识别出错: {e}，默认按CTA处理")
        report_type = "CTA"
    
    # 初始化top_data
    top_data = {}
    
    # 🔍 调试信息：显示初始表格状态
    print(f"🔍 初始formatted_table行数: {len(formatted_table)}")
    if len(formatted_table) > 0:
        print(f"🔍 初始表格前5行名称: {formatted_table['名称'].head().tolist()}")
    
    # ==============================================================
    # 第二步：根据报告类型执行不同的处理逻辑
    # ==============================================================
    
    print(f"🔍 即将进入处理分支: {report_type}")
    
    if report_type == "CTA":
        print("🔍 按冠脉CTA报告处理...")
        
        # CTA报告的顶部信息提取
        header_data = [["冠状动脉钙化总积分", "LM", "LAD", "LCX", "RCA"]]
        
        # 提取顶部基本信息
        for row in header_data:
            input_prompt = FILL_IN_FORM_PROMPT.format(ocr_text=ocr, key_info=row)
            try:
                completion = qwen.chat.completions.create(
                    model=medical_model,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': input_prompt}
                    ]
                )
                text = completion.choices[0].message.content
                tmp = safe_json_load(text)
                if tmp is not None:
                    for k in tmp:
                        top_data[k] = tmp[k] if tmp[k] != "NO" else ""
            except Exception as e:
                print(f"⚠️ CTA顶部信息提取失败: {e}")
        
        # 提取CTA专用的分类信息
        cta_prompts = [
            FILLIN_PROMPT_2.format(ocr_text=ocr),  # 冠状动脉起源、走形及终止
            FILLIN_PROMPT_3.format(ocr_text=ocr),  # 冠脉优势型
            FILLIN_PROMPT_4.format(ocr_text=ocr)   # 异常描述
        ]
        
        for input_prompt in cta_prompts:
            try:
                completion = qwen.chat.completions.create(
                    model=medical_model,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': input_prompt}
                    ]
                )
                text = completion.choices[0].message.content
                tmp = safe_json_load(text)
                if tmp is not None and 'key_name' in tmp and 'result' in tmp:
                    top_data[tmp['key_name']] = tmp['result']
            except Exception as e:
                print(f"⚠️ CTA分类信息提取失败: {e}")

        # 并行处理CTA的55个冠脉节段
        print("🔄 开始并行处理冠脉节段...")

        import concurrent.futures
        from functools import partial
        
        def process_location(location, ocr, qwen, row_index, system_prompt, model_name):
            """处理单个冠脉节段的函数，用于并行执行"""
            if location not in row_index:
                return None, None, None
            
            ridx = row_index[location]
            
            input_prompt = FILLIN_PROMPT_5.format(ocr_text=ocr, location=location)
            
            # 实现重试机制（指数退避）
            max_retries = 3
            attempt = 0
            
            while attempt < max_retries:
                try:
                    completion = qwen.chat.completions.create(
                        model=model_name,
                        messages=[
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': input_prompt}
                        ]
                    )
                    text = completion.choices[0].message.content
                    tmp = safe_json_load(text)
                    return location, ridx, tmp
                except Exception as e:
                    attempt += 1
                    if attempt < max_retries:
                        # 指数退避：等待时间为 2^attempt 秒，最大等待32秒
                        wait_time = min(2 ** attempt, 32)
                        import time
                        time.sleep(wait_time)
                    else:
                        print(f"❌ {location} 处理失败: {e}")
                        return location, ridx, None
            
            return location, ridx, None
        
        # 获取所有需要处理的位置
        locations_to_process = []
        for i in range(len(formatted_table)):
            location = formatted_table.iloc[i]["名称"]
            if location in row_index:
                locations_to_process.append(location)
        
        # 并行处理
        process_func = partial(process_location, ocr=ocr, qwen=qwen, row_index=row_index, system_prompt=SYSTEM_PROMPT, model_name=medical_model)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(process_func, locations_to_process))
        
        # 更新表格 - 只更新CTA相关的字段
        updatable_columns = ["类型", "症状", "数值", "单位"]
        all_extracted_data = {}  # 收集所有提取到的数据用于后续动态添加
        
        for location, ridx, tmp in results:
            if tmp is not None:
                # 检查是否有实际数据（不是默认的"-"值）
                has_real_data = False
                for col in updatable_columns:
                    value = tmp.get(col, "")
                    if value == "NO":
                        value = ""
                    if value and value != "-":
                        formatted_table.at[ridx, col] = value
                        if col == "数值" and value != "-":
                            has_real_data = True
                            all_extracted_data[location] = value
                
                if has_real_data:
                    print(f"✅ CTA匹配成功: {location}")
        
        # 收集所有可能被遗漏的CTA数据，添加到表格底部
        print("📊 检查是否有遗漏的CTA数据...")
        try:
            # 使用通用提取prompt找出可能遗漏的冠脉相关数据
            cta_general_prompt = f"""
我将给你一段冠脉CTA诊断报告经过OCR提取之后的文本。请从中提取所有具体的冠脉节段信息和测量数值，特别是那些可能没有包含在以下已知项目中的数据：

**已知项目：**
{', '.join(all_extracted_data.keys())}

**医疗诊断报告：**
{ocr}

**提取规则：**
1. 重点关注冠脉节段（如"左主干"、"前降支"、"回旋支"、"右冠"的各个分段）
2. 提取斑块、狭窄、钙化等相关信息
3. 包含具体的狭窄程度数值

**返回格式：**
请以JSON格式返回所有冠脉相关数据：
{{
"冠脉节段或测量项目": "相关信息或数值",
"冠脉节段或测量项目": "相关信息或数值",
...
}}

**只返回JSON，不要输出其他内容。**
"""
            
            completion = qwen.chat.completions.create(
                model=medical_model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': cta_general_prompt}
                ]
            )
            cta_data_text = completion.choices[0].message.content
            cta_data = safe_json_load(cta_data_text)
            
            if cta_data:
                # 找出未匹配的数据
                unmatched_data = []
                for key, value in cta_data.items():
                    # 检查这个数据是否已经被匹配过
                    was_matched = any(str(all_extracted_data.get(existing_key, "")) == str(value) 
                                    for existing_key in all_extracted_data.keys())
                    
                    # 同时检查key是否已经在现有表格的名称列中
                    key_exists = any(key in str(formatted_table.iloc[i]["名称"]) or 
                                   str(formatted_table.iloc[i]["名称"]) in key 
                                   for i in range(len(formatted_table)))
                    
                    if not was_matched and not key_exists and value and value != "-":
                        unmatched_data.append((key, value))
                
                # 将未匹配的数据添加到表格底部
                if unmatched_data:
                    print(f"📋 发现 {len(unmatched_data)} 个未匹配的CTA数据，添加到表格底部...")
                    
                    import pandas as pd
                    for key, value in unmatched_data:
                        # 分离数值和单位
                        pure_value, extracted_unit = separate_value_and_unit(str(value))
                        
                        new_row = {
                            "名称": key,
                            "英文": "",  # 不推测英文，保持空白
                            "斑块种类": "",
                            "类型": "",
                            "症状": "",
                            "数值": pure_value,
                            "单位": extracted_unit,  # 自动提取的单位
                            "狭窄程度": "",
                            "闭塞": ""
                        }
                        
                        # 使用pd.concat添加新行到表格最前面
                        new_row_df = pd.DataFrame([new_row])
                        formatted_table = pd.concat([new_row_df, formatted_table], ignore_index=True)
                        
                        print(f"➕ 添加新行: {key} = {value}")
                        
        except Exception as e:
            print(f"⚠️ CTA补充提取失败: {e}")
        
        print("✅ CTA报告处理完成")
    
    elif report_type == "Ultrasound":
        print("🫀 按心脏超声报告处理...")
        
        # 先抽取头部信息（姓名/性别/年龄/设备/所见/提示等），缺失留空
        try:
            from medical_agent.prompts import ULTRASOUND_HEADER_PROMPT
            header_prompt = ULTRASOUND_HEADER_PROMPT.format(ocr_text=ocr)
            completion = qwen.chat.completions.create(
                model=medical_model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': header_prompt}
                ]
            )
            text = completion.choices[0].message.content
            header_json = safe_json_load(text) or {}
            if isinstance(header_json, dict):
                for k, v in header_json.items():
                    top_data[k] = v or ""
        except Exception as e:
            print(f"⚠️ 超声头部信息提取失败: {e}")
        
        # 关键测量值不再在此处用LLM抽取，改为在表格完成后从表格中回填到 top_data
        # （LVEF, LVEDD, LVESD, IVSd, LVPWd, E/A, e′, a′）

        # 保留原有“异常描述”提取作为补充（若 header_json 未给到）
        try:
            if not top_data.get('异常描述'):
                input_4 = FILLIN_PROMPT_4.format(ocr_text=ocr)
                completion = qwen.chat.completions.create(
                    model=medical_model,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': input_4}
                    ]
                )
                text = completion.choices[0].message.content
                tmp = safe_json_load(text)
                if tmp is not None and 'key_name' in tmp and 'result' in tmp:
                    top_data[tmp['key_name']] = tmp['result']
        except Exception as e:
            print(f"⚠️ 超声异常描述提取失败: {e}")
        
        # 并行处理超声的55个测量项目
        print("🔄 开始并行处理超声测量项目...")
        
        import concurrent.futures
        from functools import partial
        
        # 先从全文抽取所有候选“项目→数值(可含单位)”
        candidates = {}
        try:
            from medical_agent.prompts import ULTRASOUND_ALL_MEASUREMENTS_PROMPT
            cand_prompt = ULTRASOUND_ALL_MEASUREMENTS_PROMPT.format(ocr_text=ocr)
            completion = qwen.chat.completions.create(
                model=medical_model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': cand_prompt}
                ]
            )
            cand_text = completion.choices[0].message.content
            cand_json = safe_json_load(cand_text) or {}
            if isinstance(cand_json, dict):
                candidates = cand_json
        except Exception as e:
            print(f"⚠️ 候选测量抽取失败: {e}")
        
        # 准备标准表候选集合（名称与英文）
        std_name_to_ridx = {}
        std_choices = []
        for i in range(len(formatted_table)):
            cn_name = str(formatted_table.iloc[i]["名称"]).strip()
            std_name_to_ridx[cn_name.lower()] = i
            std_choices.append(cn_name)
            eng = str(formatted_table.iloc[i].get("英文", "") or "").strip()
            if eng:
                std_name_to_ridx[eng.lower()] = i
                std_choices.append(eng)
        
        # 知识库索引
        try:
            from medical_agent.normalizer import _load_kb, _build_alias_index
            kb = _load_kb()
            alias_to_canonical, canonical_meta = _build_alias_index(kb)
        except Exception as _e:
            alias_to_canonical, canonical_meta = {}, {}
        
        free_rows = []
        consumed_keys = set()
        llm_cache: Dict[str, str] = {}
        
        def update_std_row_by_ridx(ridx: int, raw_value: str):
            pure_value, extracted_unit = separate_value_and_unit(str(raw_value))
            if pure_value and pure_value != "-":
                formatted_table.at[ridx, "数值"] = pure_value
                if extracted_unit:
                    cur_unit = str(formatted_table.at[ridx, "单位"]) if formatted_table.at[ridx, "单位"] is not None else ""
                    if not cur_unit.strip():
                        formatted_table.at[ridx, "单位"] = extracted_unit
        
        # 阶段1：标准表（精确 -> rapidfuzz -> 灰区Qwen校验）
        for key, raw_value in candidates.items():
            q = _preclean_name(key)
            # exact (大小写不敏感)
            if q.lower() in std_name_to_ridx:
                update_std_row_by_ridx(std_name_to_ridx[q.lower()], raw_value)
                consumed_keys.add(key)
                continue
            # rapidfuzz
            best = process.extractOne(q, std_choices, scorer=fuzz.WRatio)
            if best:
                choice, score, _ = best
                if score >= 95:
                    ridx = std_name_to_ridx.get(choice.lower())
                    if ridx is not None:
                        update_std_row_by_ridx(ridx, raw_value)
                        consumed_keys.add(key)
                        continue
                elif 80 <= score < 95:
                    # 灰区：取topK候选交给Qwen复核
                    topk = _rapid_topk(q, std_choices, k=8)
                    cands = []
                    for c, s, _ in topk:
                        ridx = std_name_to_ridx.get(c.lower())
                        if ridx is None:
                            continue
                        eng = str(formatted_table.iloc[ridx].get("英文", "") or "").strip()
                        cands.append({"exact_name": str(formatted_table.iloc[ridx]["名称"]).strip(), "aliases": [eng] if eng else []})
                    cache_key = f"std::{q}::{json.dumps(cands, ensure_ascii=False)}"
                    match = llm_cache.get(cache_key)
                    if match is None:
                        match = _ask_qwen_alias(qwen, medical_model, key, cands)
                        llm_cache[cache_key] = match or ""
                    if match and match != "no_match":
                        ridx = std_name_to_ridx.get(match.lower())
                        # 若直接中文名未命中，再遍历找名称匹配
                        if ridx is None:
                            for i in range(len(formatted_table)):
                                if str(formatted_table.iloc[i]["名称"]).strip() == match:
                                    ridx = i
                                    break
                        if ridx is not None:
                            update_std_row_by_ridx(ridx, raw_value)
                            consumed_keys.add(key)
                            continue
        
        # 阶段2：KB（rapidfuzz -> 灰区Qwen校验） -> 自由行
        kb_alias_keys = list(alias_to_canonical.keys()) if alias_to_canonical else []
        for key, raw_value in candidates.items():
            if key in consumed_keys:
                continue
            q = _preclean_name(key).lower()
            canonical = ""
            if q in alias_to_canonical:
                canonical = alias_to_canonical[q]
            elif kb_alias_keys:
                best = process.extractOne(q, kb_alias_keys, scorer=fuzz.WRatio)
                if best:
                    cand_key, score, _ = best
                    if score >= 90:
                        canonical = alias_to_canonical[cand_key]
                    elif 80 <= score < 90:
                        # 灰区：取topK候选交给Qwen
                        topk = process.extract(q, kb_alias_keys, scorer=fuzz.WRatio, limit=8)
                        # 归并成 canonical 候选并去重
                        canon_set = []
                        seen = set()
                        for ck, s, _ in topk:
                            cn = alias_to_canonical.get(ck, "")
                            if cn and cn not in seen:
                                meta = canonical_meta.get(cn, {})
                                aliases = []
                                abbr = str(meta.get("测量值简写", "") or "").strip()
                                eng = str(meta.get("测量值英文", "") or "").strip()
                                if abbr: aliases.append(abbr)
                                if eng: aliases.append(eng)
                                alias_field = meta.get("别名", []) or []
                                if isinstance(alias_field, str):
                                    alias_field = [a.strip() for a in alias_field.split(";") if a.strip()]
                                aliases.extend(alias_field)
                                canon_set.append({"exact_name": cn, "aliases": aliases})
                                seen.add(cn)
                        cache_key = f"kb::{q}::{json.dumps(canon_set, ensure_ascii=False)}"
                        match = llm_cache.get(cache_key)
                        if match is None:
                            match = _ask_qwen_alias(qwen, medical_model, key, canon_set)
                            llm_cache[cache_key] = match or ""
                        if match and match != "no_match":
                            canonical = match
            
            if canonical:
                meta = canonical_meta.get(canonical, {})
                abbr = str(meta.get("测量值简写", "") or "").strip()
                english = str(meta.get("测量值英文", "") or "").strip()
                unit_std = str(meta.get("单位", "") or "").strip()
                pure_value, extracted_unit = separate_value_and_unit(str(raw_value))
                final_unit = extracted_unit or unit_std
                std_name = f"{canonical}({abbr})" if abbr else canonical
                free_rows.append({
                    "名称": std_name,
                    "英文": english,
                            "类型": "",
                            "症状": "",
                            "数值": pure_value,
                    "单位": final_unit
                })
                consumed_keys.add(key)
        
        # 阶段3：兜底自由行
        for key, raw_value in candidates.items():
            if key in consumed_keys:
                continue
            pure_value, extracted_unit = separate_value_and_unit(str(raw_value))
            free_rows.append({
                "名称": str(key).strip(),
                "英文": "",
                "类型": "",
                "症状": "",
                "数值": pure_value,
                "单位": extracted_unit
            })
        
        # 将自由行插入到表格前部
        if free_rows:
            import pandas as pd
            free_df = pd.DataFrame(free_rows)
            formatted_table = pd.concat([free_df, formatted_table], ignore_index=True)
        
        print("✅ 超声报告处理完成")
    
    else:
        print(f"⚠️ 未知报告类型: {report_type}，跳过处理")
    
    # ==============================================================
    # 第三步：更新状态中的表格并保存结果
    # ==============================================================
    
    # 如果状态中有处理时间的起点，打印处理时间
    if 'process_start_time' in state['context']:
        end_timer_and_print(state['context']['process_start_time'], 
                          state['context'].get('current_file_name', 'unknown'), 
                          state['context'].get('file_type', 'file'))
    
    print("💾 保存结果...")
    
    # 更新状态中的表格（可能已经被动态扩展）
    try:
        # 在保存前进行基于知识库的归一化
        from medical_agent.normalizer import normalize_table_with_kb
        formatted_table = normalize_table_with_kb(formatted_table)
    except Exception as _e:
        # 归一化失败不影响主流程
        print(f"⚠️ 归一化步骤跳过: {_e}")
    state['formatted_table'] = formatted_table
    
    # 从最终表格中回填关键测量值到 top_data（逐个击破：每个 key 单独询问 LLM → 再从表内取值）
    try:
        from medical_agent.prompts import ULTRASOUND_KEY_NAME_PICK_PROMPT
        import json as _json

        TARGET_KEYS = ["LVEF", "LVEDD", "LVESD", "IVSd", "LVPWd", "E/A", "e′", "E/e′", "a′"]
        # 候选名称：仅取“名称”列的非空去重值
        candidate_names = []
        seen = set()
        for i in range(len(formatted_table)):
            nm = str(formatted_table.iloc[i].get("名称", "") or "").strip()
            if nm and nm not in seen:
                candidate_names.append(nm)
                seen.add(nm)
        # 快速索引：名称 -> 值字符串
        name_to_val = {}
        for i in range(len(formatted_table)):
            nm = str(formatted_table.iloc[i].get("名称", "") or "").strip()
            val = str(formatted_table.iloc[i].get("数值", "") or "").strip()
            unit = str(formatted_table.iloc[i].get("单位", "") or "").strip()
            if nm and val and val not in ("-", "NO"):
                name_to_val[nm] = f"{val}{unit}" if unit else val

        for key in TARGET_KEYS:
            try:
                payload = ULTRASOUND_KEY_NAME_PICK_PROMPT.format(
                    target_key=key,
                    candidate_names_json=_json.dumps(candidate_names, ensure_ascii=False)
                )
                completion = qwen.chat.completions.create(
                    model=medical_model,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': payload}
                    ]
                )
                text = completion.choices[0].message.content
                res = safe_json_load(text) or {}
                match_name = ""
                if isinstance(res, dict):
                    match_name = str(res.get("match", "") or "").strip()
                if match_name:
                    top_data[key] = name_to_val.get(match_name, "")
                else:
                    # 无匹配则置空（确保不保留旧值）
                    top_data[key] = ""
            except Exception as _inner:
                print(f"⚠️ 关键项 {key} 回填失败: {_inner}")
                top_data[key] = ""
    except Exception as _e:
        print(f"⚠️ 顶部关键测量值回填失败: {_e}")

    save_df_to_cache(formatted_table, "qwen_cache")

    # 加载并显示结果
    df = load_df_from_cache("qwen_cache")
    state['context']['df'] = df
    
    print("🎯 显示结果...")
    show_popup_with_df(df, top_data)

    print("✨ 智能结构化提取完成！")
    return state


            

# Define the response generation node
def create_response_node(state: AgentState):
    """Create a node for handling user input and generating responses."""
    # Get the last user message
    last_message = next((m for m in reversed(state["messages"]) if m["role"] == "user"), None)
    
    if not last_message:
        return state
        
    # Get content and content type
    content = last_message["content"]
    content_type = last_message.get("content_type", "text")
    
    # Get response from LLM using state's llm
    # result = state["gpt"].invoke([system_message, user_message])
    completion = call_qwen_vl_api(state)
    
    # Add the response to messages
    state["messages"].append({
        "role": "assistant",
        "content": completion,
        "content_type": "text"
    })
    
    return state
    
def show_results(state: AgentState):
    """Show the results of the agent's response."""
    # Get the last assistant message
    last_assistant_message = next((m for m in reversed(state["messages"]) if m["role"] == "assistant"), None)

    if last_assistant_message:
        print("Agent Response:")
        print(last_assistant_message["content"])
    else:
        print("No response from the agent.")
    return state

def _preclean_name(text: str) -> str:
    import re
    if not text:
        return ""
    s = str(text)
    s = re.sub(r"[\(（][^\)）]*[\)）]", "", s)
    s = re.sub(r"[%：:，,。·/\\]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _rapid_topk(query: str, choices: List[str], k: int = 10) -> List[tuple]:
    if not choices:
        return []
    return process.extract(_preclean_name(query), choices, scorer=fuzz.WRatio, limit=k)


def _ask_qwen_alias(qwen_client, model_name: str, query: str, candidates: List[Dict[str, Any]]) -> str:
    from medical_agent.prompts import ALIAS_VALIDATION_PROMPT
    try:
        payload = ALIAS_VALIDATION_PROMPT.format(
            query=query,
            candidates_json=json.dumps(candidates, ensure_ascii=False)
        )
        completion = qwen_client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': payload}
            ]
        )
        text = completion.choices[0].message.content
        res = safe_json_load(text) or {}
        match = res.get("match", "") if isinstance(res, dict) else ""
        return str(match)
    except Exception:
        return ""

# Build the complete agent
def build_medical_agent():
    """Build a simplified medical agent with a single node."""
    
    graph = StateGraph(AgentState)
    
    graph.add_node("init_llms", init_llms)
    graph.add_node("input_node", create_input_node)
    graph.add_node("ocr_node", ocr_node)
    graph.add_node("response_node", create_response_node)
    graph.add_node("show_results", show_results)
    graph.add_node("fill_form_node", fill_form_node)
    # Set the entry point and edge
    graph.set_entry_point("init_llms")
    graph.add_edge("init_llms", "input_node")
    graph.add_edge("input_node", "ocr_node")
    graph.add_edge("ocr_node", "fill_form_node")
    # graph.add_edge("ocr_node", "response_node")
    # graph.add_edge("response_node", "show_results")
    # Compile the graph
    return graph.compile()