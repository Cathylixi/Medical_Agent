# AI提取的原始数据（草稿）
ai_result = [
    {"冠脉节段": "左主干近段(pLM)", "狭窄程度": "局限性狭窄", "数值": 45},
    {"冠脉节段": "左前降支近段(pLAD)", "狭窄程度": "阶段性狭窄", "数值": 30},
]

# 医院标准表的映射（部分）
standard_table = {
    ("左主干近段(pLM)", "狭窄程度"): {"编号": 201, "测量值名称": "左主干狭窄程度", "单位": "%"},
    ("左前降支近段(pLAD)", "狭窄程度"): {"编号": 202, "测量值名称": "左前降支狭窄程度", "单位": "%"},
}

# 转换成医院标准格式
standard_result = []
for row in ai_result:
    key = (row["冠脉节段"], "狭窄程度")
    if key in standard_table:
        std = standard_table[key]
        standard_result.append({
            "测量值编号": std["编号"],
            "测量值名称": std["测量值名称"],
            "单位": std["单位"],
            "数值": row["数值"],
            "患者ID": "123456",
            "检查时间": "2024-06-01"
        })

# 打印最终结果
for item in standard_result:
    print(item) 