import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from agent import AgentState, init_llms, fill_form_node
from utils import save_df_to_cache, save_ocr_result, start_timer, end_timer_and_print
import pandas as pd

# Load environment variables
load_dotenv()

def extract_from_text(text_input: str, output_name: str = "text_extraction_result") -> bool:
    """
    从自然语言文本中提取结构化信息
    
    Args:
        text_input (str): 输入的自然语言文本
        output_name (str): 输出文件名（不含扩展名）
        
    Returns:
        bool: 处理是否成功
    """
    # 开始计时
    start_time = start_timer()
    
    try:
        print(f"开始处理文本（长度：{len(text_input)}字符）")
        
        # 1. 创建初始状态
        state = AgentState()
        state = init_llms(state)
        
        # 2. 直接将文本作为OCR结果放入状态
        state['context'] = {'ocr': text_input}
        
        # 保存OCR结果（文本输入的情况下，原文本就是OCR结果）
        save_ocr_result(text_input, output_name, "text")
        
        print("开始结构化提取...")
        
        # 3. 运行结构化提取
        state = fill_form_node(state)
        
        # 4. 检查结果并保存
        if 'formatted_table' in state:
            df = state['formatted_table']
            save_df_to_cache(df, output_name)
            print(f"\n✅ 文本结构化完成，结果已保存到 {output_name}.parquet")
            
            end_timer_and_print(start_time, output_name, "文本输入")
            return True
        else:
            print("⚠️ 结构化提取失败")
            end_timer_and_print(start_time, output_name, "文本输入")
            return False
            
    except Exception as e:
        print(f"❌ 处理文本时出错: {e}")
        end_timer_and_print(start_time, output_name, "文本输入")
        return False

def main():
    """
    主函数：医疗报告自然语言文本结构化提取
    """
    print("🏥 医疗报告自然语言文本结构化提取工具")
    print("=" * 60)
    
    # 检查是否在自动测试模式
    if os.environ.get('AUTO_RUN', '0') == '1':
        # 自动测试模式：使用预设文本
        test_text = """CT检查报告单
患者姓名：李明  性别：男  年龄：58岁  住院号：2023041567
检查项目：冠状动脉CTA检查
检查所见：
冠状动脉呈左优势型。左主干起源于左窦，右冠状动脉起源于右窦。
左主干管壁可见混合斑块，管腔中度狭窄约45%。
左前降支近段管壁可见非钙化斑块，管腔轻度狭窄约30%；中段管壁可见钙化斑块，管腔中度狭窄约55%；远段未见明显狭窄。
第一对角支未见斑块及明显狭窄。第二对角支管壁可见钙化斑块，管腔轻度狭窄约25%。
左回旋支近段管壁可见钙化斑块，管腔重度狭窄约75%；中段、远段未见明显狭窄。
第一钝缘支管壁可见非钙化斑块，管腔中度狭窄约40%。第二钝缘支未见斑块及明显狭窄。
右冠状动脉近段管壁可见混合斑块，管腔重度狭窄约80%；中段管壁可见钙化斑块，管腔中度狭窄约50%；远段未见明显狭窄。
心脏各腔室大小正常，心肌未见异常密度影。
印象：
冠状动脉CTA：1.左主干管壁混合斑块，管腔中度狭窄。2.左前降支近段非钙化斑块，管腔轻度狭窄；中段钙化斑块，管腔中度狭窄。3.左回旋支近段钙化斑块，管腔重度狭窄。4.右冠状动脉近段混合斑块，管腔重度狭窄；中段钙化斑块，管腔中度狭窄。
检查日期：2023-04-15  报告医师：王医师
        """
        
        print("自动测试模式：使用预设的测试文本")
        print("预测结果：")
        print("基于输入文本，我预测提取的结构化数据应该包含：")
        print("- 冠脉优势型：左优势型")
        print("- 左主干(LM)：混合斑块，中度狭窄约45%")
        print("- 左前降支近段(pLAD)：软斑块，轻度狭窄约30%")
        print("- 左前降支中段(mLAD)：硬斑块，中度狭窄约55%")
        print("- 左回旋支近段(pLCX)：硬斑块，重度狭窄约75%")
        print("- 右冠近段(pRCA)：混合斑块，重度狭窄约80%")
        print("- 右冠中段(mRCA)：硬斑块，中度狭窄约50%")
        print("- 第二对角支(D2)：硬斑块，轻度狭窄约25%")
        print("- 第一钝缘支(OM1)：软斑块，中度狭窄约40%")
        print("\n现在开始实际测试...")
        print()
        
        user_text = test_text
        output_name = "test_natural_language_extraction"
        
    else:
        # 交互模式：让用户输入文本
        print("请选择输入方式：")
        print("1. 直接在终端输入文本")
        print("2. 从文件读取文本")
        print("3. 使用预设的测试文本")
        
        choice = input("请输入选项 (1/2/3): ").strip()
        
        if choice == "1":
            print("\n请输入医疗报告文本（输入完成后按 Ctrl+D (Mac/Linux) 或 Ctrl+Z (Windows) 结束）：")
            print("-" * 40)
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            user_text = "\n".join(lines)
            
        elif choice == "2":
            # 处理data/test_text/目录下的所有txt文件
            test_dir = "data/test_text"
            if not os.path.exists(test_dir):
                print(f"❌ 测试文件目录不存在: {test_dir}")
                return
                
            txt_files = [f for f in os.listdir(test_dir) if f.endswith('.txt')]
            if not txt_files:
                print(f"❌ 在 {test_dir} 目录下未找到任何txt文件")
                return
                
            print(f"\n📁 找到以下txt文件：")
            for i, file in enumerate(txt_files, 1):
                print(f"{i}. {file}")
            print("\n开始批量处理...")
            
            # 处理每个文件
            for i, file in enumerate(txt_files, 1):
                file_path = os.path.join(test_dir, file)
                print(f"\n{'='*60}")
                print(f"处理第 {i}/{len(txt_files)} 个文件: {file}")
                print(f"{'='*60}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        user_text = f.read()
                    print(f"✅ 成功读取文件: {file}")
                    
                    # 使用文件名（不含扩展名）作为输出文件名
                    output_name = os.path.splitext(file)[0]
                    
                    # 处理文本（时间统计已在函数内部处理）
                    success = extract_from_text(user_text, output_name)
                    if success:
                        print(f"✅ 文件 {file} 处理完成")
                    else:
                        print(f"❌ 文件 {file} 处理失败")
                        
                except Exception as e:
                    print(f"❌ 读取或处理文件 {file} 时出错: {e}")
                    continue
                    
            print("\n📊 批量处理完成!")
            return
                
        elif choice == "3":
            user_text = """CT检查报告单
患者姓名：李明  性别：男  年龄：58岁  住院号：2023041567
检查项目：冠状动脉CTA检查
检查所见：
冠状动脉呈左优势型。左主干起源于左窦，右冠状动脉起源于右窦。
左主干管壁可见混合斑块，管腔中度狭窄约45%。
左前降支近段管壁可见非钙化斑块，管腔轻度狭窄约30%；中段管壁可见钙化斑块，管腔中度狭窄约55%；远段未见明显狭窄。
第一对角支未见斑块及明显狭窄。第二对角支管壁可见钙化斑块，管腔轻度狭窄约25%。
左回旋支近段管壁可见钙化斑块，管腔重度狭窄约75%；中段、远段未见明显狭窄。
第一钝缘支管壁可见非钙化斑块，管腔中度狭窄约40%。第二钝缘支未见斑块及明显狭窄。
右冠状动脉近段管壁可见混合斑块，管腔重度狭窄约80%；中段管壁可见钙化斑块，管腔中度狭窄约50%；远段未见明显狭窄。
心脏各腔室大小正常，心肌未见异常密度影。
印象：
冠状动脉CTA：1.左主干管壁混合斑块，管腔中度狭窄。2.左前降支近段非钙化斑块，管腔轻度狭窄；中段钙化斑块，管腔中度狭窄。3.左回旋支近段钙化斑块，管腔重度狭窄。4.右冠状动脉近段混合斑块，管腔重度狭窄；中段钙化斑块，管腔中度狭窄。
检查日期：2023-04-15  报告医师：王医师
            """
            print("使用预设测试文本")
            
        else:
            print("无效选项，退出程序")
            return
        
        # 让用户指定输出文件名（仅在非批处理模式下）
        if choice != "2":  # 如果不是文件批处理模式
            output_name = input("请输入输出文件名（不含扩展名，直接回车使用默认名称）: ").strip()
            if not output_name:
                output_name = "text_extraction_result"
    
    # 检查文本是否为空
    if not user_text.strip():
        print("❌ 输入文本为空，请重新运行程序")
        return
    
    # 开始处理
    success = extract_from_text(user_text, output_name)
    
    if success:
        print("\n🎉 自然语言文本结构化提取完成！")
        print("💡 提示：可以使用 GUI 工具查看详细结果")
        if os.environ.get('AUTO_RUN', '0') == '1':
            print()
            print("📝 对比分析：")
            print("请对比上面的预测结果和实际提取结果，看看是否匹配！")
    else:
        print("\n⚠️ 处理未成功完成，请检查错误信息")

if __name__ == "__main__":
    main() 