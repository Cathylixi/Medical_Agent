import pandas as pd

# 不再维护固定ROW_INDEX；提供动态行索引生成功能
ROW_INDEX = {}


def create_formatted_df():
    """
    创建格式化的DataFrame
    完全从 data/标准测量表.xlsx 动态加载，便于随时修改标准表

    Returns:
        pd.DataFrame: 格式化后的数据框
    """
    # 定义列（移除：斑块种类、狭窄程度、闭塞）
    columns = ["名称", "英文", "类型", "症状", "数值", "单位"]

    # 动态读取标准测量表.xlsx
    try:
        standard_df = pd.read_excel('data/标准测量表.xlsx')
        
        # 🔍 调试信息：显示标准测量表内容
        print(f"🔍 标准测量表.xlsx总行数: {len(standard_df)}")
        if len(standard_df) > 0:
            print(f"🔍 标准测量表列名: {standard_df.columns.tolist()}")
            print(f"🔍 标准测量表前5行中文名称: {standard_df['中文名称'].head().tolist()}")
        
        # 构建动态行数据
        dynamic_rows = []
        for _, row in standard_df.iterrows():
            if pd.isna(row.get('中文名称')) or str(row.get('中文名称')).strip() in ('', 'left'):
                continue
            cn = str(row['中文名称']).strip()
            abbr = str(row.get('测量值简写', '') or '').strip()
            name = f"{cn}({abbr})" if abbr else cn
            english = str(row.get('测量值名称', '') or '').strip()
            unit = str(row.get('单位', '') or '').strip()
            row_data = [
                name,           # 名称
                english,        # 英文
                "",            # 类型
                "",            # 症状
                "",            # 数值
                unit            # 单位
            ]
            dynamic_rows.append(row_data)
            
            # 🔍 调试信息：显示是否包含冠脉相关词汇
            if any(keyword in cn for keyword in ['冠', '主干', '前降支', '回旋支']):
                print(f"🔍 发现冠脉相关项: {name}")
                
        print(f"✅ 从标准测量表.xlsx成功读取 {len(dynamic_rows)} 行数据")
    except Exception as e:
        print(f"⚠️ 读取标准测量表.xlsx失败: {e}")
        print("   使用空的动态数据")
        dynamic_rows = []

    # 构建DataFrame
    df = pd.DataFrame(dynamic_rows, columns=columns)
    return df


def get_dynamic_row_index():
    """
    基于当前标准测量表动态生成行索引映射
    """
    df = create_formatted_df()
    row_index = {}
    for idx, row in df.iterrows():
        row_index[row['名称']] = idx
    return row_index
