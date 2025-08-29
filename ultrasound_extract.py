import os
import re
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from agent import AgentState, init_llms
from utils import save_df_to_cache

# Load environment variables
load_dotenv()

# 超声检查的测量项目提取Prompt
ULTRASOUND_EXTRACT_PROMPT = """
我将给你一段超声诊断报告经过OCR提取之后的文本。请从中提取所有具体的测量数值和参数。

**医疗诊断报告：**
{ocr_text}

请提取所有提到的测量值，包括但不限于：
- 心脏各腔室尺寸（如左房、左室、右房、右室等）
- 血管尺寸（如主动脉、肺动脉等）
- 壁厚度（如室间隔厚度、后壁厚度等）
- 功能参数（如EF值、FS值等）
- 血流速度和压力梯度

**返回格式：**
请返回JSON格式，每个测量项目包含：项目名称、数值、单位

{{
"measurements": [
    {{"name": "测量项目名称", "value": "数值", "unit": "单位", "description": "详细描述"}},
    {{"name": "测量项目名称", "value": "数值", "unit": "单位", "description": "详细描述"}},
    ...
]
}}

**输出：**
"""

def extract_ultrasound_measurements(text_input: str, output_name: str = "ultrasound_extraction") -> bool:
    """
    从超声报告文本中提取测量数据
    
    Args:
        text_input (str): 输入的超声报告文本
        output_name (str): 输出文件名
        
    Returns:
        bool: 处理是否成功
    """
    try:
        print(f"开始处理超声报告文本（长度：{len(text_input)}字符）")
        print("=" * 60)
        
        # 1. 创建初始状态
        state = AgentState()
        state = init_llms(state)
        
        # 2. 使用专门的超声提取prompt
        prompt = ULTRASOUND_EXTRACT_PROMPT.format(ocr_text=text_input)
        
        print("开始超声测量数据提取...")
        
        # 3. 调用AI进行结构化提取
        qwen = state["qwen"]
        completion = qwen.chat.completions.create(
            model="qwen-max-0125",
            messages=[
                {'role': 'system', 'content': '你是一个医疗助手，专门从超声报告中提取测量数据。'},
                {'role': 'user', 'content': prompt}
            ]
        )
        
        result_text = completion.choices[0].message.content
        print(f"AI返回结果：\n{result_text}")
        
        # 4. 解析结果并转换为DataFrame
        import json
        try:
            result_json = json.loads(result_text)
            measurements = result_json.get('measurements', [])
            
            # 转换为DataFrame
            df = pd.DataFrame(measurements)
            
            if not df.empty:
                print(f"\n✅ 成功提取 {len(measurements)} 个测量项目")
                print("\n📊 提取的测量数据：")
                print("-" * 80)
                print(df.to_string(index=False))
                print("-" * 80)
                
                # 保存结果
                save_df_to_cache(df, output_name)
                print(f"\n💾 结果已保存到 {output_name}.parquet")
                
                return True
            else:
                print("⚠️ 未提取到任何测量数据")
                return False
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print("尝试使用正则表达式提取数值...")
            
            # 备用方案：使用正则表达式提取数值
            measurements = extract_numbers_with_regex(text_input)
            if measurements:
                df = pd.DataFrame(measurements)
                save_df_to_cache(df, output_name)
                print(f"✅ 使用正则表达式成功提取 {len(measurements)} 个数值")
                return True
            else:
                return False
            
    except Exception as e:
        print(f"❌ 处理超声报告时出错: {e}")
        return False

def extract_numbers_with_regex(text: str) -> list:
    """
    使用正则表达式从文本中提取数值和单位
    """
    measurements = []
    
    # 常见的超声测量模式
    patterns = [
        r'(\w+[^：:]*)[：:]\s*(\d+\.?\d*)\s*(mm|cm|m/s|mmHg|%)',  # 项目名：数值 单位
        r'(\w+[^约]*约)\s*(\d+\.?\d*)\s*(mm|cm|m/s|mmHg|%)',      # 项目名约 数值 单位
        r'(\w+)\s*(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*(mm|cm)',      # 项目名 数值x数值 单位
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) == 3:
                name, value, unit = match
                measurements.append({
                    "name": name.strip(),
                    "value": value,
                    "unit": unit,
                    "description": f"{name.strip()}: {value}{unit}"
                })
            elif len(match) == 4:  # x y z format
                name, value1, value2, unit = match
                measurements.append({
                    "name": name.strip(),
                    "value": f"{value1}x{value2}",
                    "unit": unit,
                    "description": f"{name.strip()}: {value1}x{value2}{unit}"
                })
    
    return measurements

if __name__ == "__main__":
    # 测试用的超声报告文本
    test_text = """彩色超声报告单
患者姓名：演示一  性别：男  年龄：60岁  病人ID：YS100001
检查描述：
双侧颈动脉管径对称，内-中膜增厚，管腔内探及多个大小不等斑块回声扁平斑块，左侧较大者位于分叉处，大小约15.6mm斑块回声扁平斑块，处后壁，大小约16.5x2.5mm斑块回声扁平斑块，斑块均延续至颈部及颈外动脉近段，双侧各段血流速正常。
双侧颈动脉球部管径对称，双侧血流速正常。
双侧颈内动脉管径正常，双侧血流速正常。
双侧颈外动脉血流速度未见明显异常。
    """
    
    print("🏥 超声报告测量数据提取测试")
    print("=" * 60)
    
    success = extract_ultrasound_measurements(test_text, "test_ultrasound_extraction")
    
    if success:
        print("\n🎉 超声报告数据提取测试完成！")
    else:
        print("\n⚠️ 测试未成功完成，请检查错误信息") 