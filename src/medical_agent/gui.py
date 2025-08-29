import tkinter as tk
from tkinter import ttk
import pandas as pd

# ---------------------------------------
# 1) Fixed coronary segments (table rows)
# ---------------------------------------
segments = [
    "左主干近段 (pLM)", "左主干中段 (mLM)", "左主干远段 (dLM)", 
    "左前降支近段 (pLAD)", "左前降支中段 (mLAD)", "左前降支远段 (dLAD)", 
    "左回旋支近段 (pLCX)", "左回旋支中段 (mLCX)", "左回旋支远段 (dLCX)",
    "右冠近段 (pRCA)", "右冠中段 (mRCA)", "右冠远段 (dRCA)",
    "第一对角支 (D1)", "第二对角支 (D2)", "中间支 (RI)",
    "第一钝缘支 (OM1)", "第二钝缘支 (OM2)", 
    "左室侧后降支 (L-PDA)", "右室侧降支 (L-PDA)",
    "左室侧后支 (L-PLB)", "右室侧后支 (R-PLB)", "右心室收缩压(RVSP)", "肺动脉收缩压(PASP)",
    "三尖瓣环收缩期位移(TAPSE)", "右心室游离壁基底段收缩期峰值速度(RV TDI s')", "右心室射血分数(RV EF)",
    "右心室面积变化分数(RV FAC)", "右心室流出道血速度度积分(RVOT VTI)", "右心室每博量(RV SV)",
    "右心室每博量指数(RV SVi)", "右心室流出道峰值速度(RVOT peak vel)", "右心室流出道峰值压差(RVOT peak PG)",
    "右心室流出道平均压差(RVOT mean PG)", "右心室流出道加速时间(RVOT AccT)", "右心室游离壁纵向应变(RVFWS)",
    "右心室整体纵向应变(RVGLS)", "二尖瓣反流峰值速度(MR peak Vel)", "二尖瓣反流血速度度积分(MR VTI)",
    "二尖瓣反流峰值压差(MR peak PG)", "二尖瓣反流dp/dt(MR dp/dt)", "二尖瓣反流血流汇聚区直径(MR VC)",
    "二尖瓣反流血流汇聚区面积(3D MR VCA)", "二尖瓣反流PISA半径(MR PISA r)", "二尖瓣反流PISA混叠速度(MR PISA aliasing vel)",
    "二尖瓣反流PISA有效瓣口面积(MR PISA EROA)", "二尖瓣反流PISA有效瓣口面积(3D MR EROA)", 
    "三尖瓣E峰速度(TV E)", "三尖瓣A峰速度(TV A)", "三尖瓣E/A比值(TV E/A)", "三尖瓣环脉冲多普勒速度(NA)",
    "三尖瓣环侧壁e'速度(TV e')", "三尖瓣E/e'比值(TV E/e')", "主肺动脉内径(mPA)", 
    "右肺动脉内径(rPA)", "左肺动脉内径(lPA)"
]

# -----------------------------------------------------
# 2) Table columns (after "冠脉节段") and dropdown setup
# -----------------------------------------------------
all_columns = ["名称", "英文", "类型", "症状", "数值", "单位"]

# ----------------------------------------------------------------
# 顶部区域：改为心脏超声头部信息 + 大段文本（所见/提示）
# ----------------------------------------------------------------

def show_popup_with_df(df: pd.DataFrame, top_data: dict):
    """
    Creates a popup window with:
      - Top fields: 心脏超声报告头部信息（姓名/性别/年龄/编号/设备/探头频率等）
      - Two large text areas: 超声所见、超声提示
      - Main table of segments vs. columns
    """
    root = tk.Tk()
    root.title("心脏超声结构化填写")
    
    # Style
    style = ttk.Style()
    style.map('TCombobox', fieldbackground=[('readonly', 'white')], 
              selectbackground=[('readonly', '#0078d7')], 
              selectforeground=[('readonly', 'white')])
    style.configure('TCombobox', background='white', foreground='black')

    # =========================
    # Top Frame (Patient Info)
    # =========================
    top_frame = ttk.Frame(root)
    top_frame.pack(padx=10, pady=5, fill="x")

    # 仅展示关键测量与大文本；去除基本信息行
    key_frame = ttk.Frame(top_frame)
    key_frame.grid(row=0, column=0, columnspan=8, sticky="we", pady=(6, 2))
    def add_key(field_label, key_name, col):
        ttk.Label(key_frame, text=field_label).grid(row=0, column=col*2, sticky="e", padx=4)
        e = ttk.Entry(key_frame, width=12)
        e.grid(row=0, column=col*2+1, padx=4)
        e.insert(0, top_data.get(key_name, ""))
        return e
    add_key("LVEF:", "LVEF", 0)
    add_key("LVEDD:", "LVEDD", 1)
    add_key("LVESD:", "LVESD", 2)
    add_key("IVSd:", "IVSd", 3)
    add_key("LVPWd:", "LVPWd", 4)
    add_key("E/A:", "E/A", 5)
    add_key("e′:", "e′", 6)
    add_key("a′:", "a′", 7)

    # Row 1: 超声所见（大文本）
    ttk.Label(top_frame, text="超声所见:").grid(row=1, column=0, sticky="ne", padx=5)
    txt_findings = tk.Text(top_frame, height=6, width=90)
    txt_findings.grid(row=1, column=1, columnspan=7, sticky="we", padx=5)
    txt_findings.insert("1.0", top_data.get("超声所见", ""))

    # Row 2: 超声提示（大文本）
    ttk.Label(top_frame, text="超声提示:").grid(row=2, column=0, sticky="ne", padx=5)
    txt_impress = tk.Text(top_frame, height=4, width=90)
    txt_impress.grid(row=2, column=1, columnspan=7, sticky="we", padx=5)
    txt_impress.insert("1.0", top_data.get("超声提示", ""))

    # =========================
    # Main Table (segments)
    # =========================
    main_table_frame = ttk.Frame(root)
    main_table_frame.pack(padx=10, pady=5, fill="both", expand=True)

    canvas = tk.Canvas(main_table_frame, width=800, height=400)
    v_scrollbar = ttk.Scrollbar(main_table_frame, orient="vertical", command=canvas.yview)
    h_scrollbar = ttk.Scrollbar(main_table_frame, orient="horizontal", command=canvas.xview)

    table_frame = ttk.Frame(canvas)

    canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    canvas.create_window((0, 0), window=table_frame, anchor="nw")

    v_scrollbar.pack(side="right", fill="y")
    h_scrollbar.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)

    all_df_columns = ["名称", "英文", "类型", "症状", "数值", "单位"]
    for j, col in enumerate(all_df_columns):
        ttk.Label(table_frame, text=col, borderwidth=1, relief="solid", width=15)\
            .grid(row=0, column=j, sticky="nsew")

    for i in range(len(df)):
        for j, col in enumerate(all_df_columns):
            raw_val = df.at[i, col] if (col in df.columns and i in df.index) else None
            value = str(raw_val) if pd.notna(raw_val) else "NONE"
            widget = ttk.Entry(table_frame, width=15)
            widget.grid(row=i+1, column=j, sticky="nsew")
            widget.insert(0, value)

    def configure_scroll_region(event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
    table_frame.bind("<Configure>", configure_scroll_region)

    def on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind("<MouseWheel>", on_mousewheel)
    canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
    canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

    root.mainloop()

# -----------------------------------------------------
# Test with dummy data (top fields + main table "NONE")
# -----------------------------------------------------
def test_gui_with_dummy_df():
    # 1) Use the real formatted DataFrame instead of dummy data
    from table_format import create_formatted_df
    df = create_formatted_df()

    # 2) Build a dict for the top fields
    top_data = {
        "姓名": "NONE",
        "性别": "NONE",
        "年龄": "NONE",
        "超声号": "NONE",
        "门诊号": "NONE",
        "住院号": "NONE",
        "床号": "NONE",
        "检查设备": "NONE",
        "检查部位": "NONE",
        "探头频率": "NONE",
        "图像质量": "NONE",
        "超声所见": "NONE",
        "超声提示": "NONE"
    }

    # 3) Show the popup
    show_popup_with_df(df, top_data)

if __name__ == "__main__":
    test_gui_with_dummy_df()
