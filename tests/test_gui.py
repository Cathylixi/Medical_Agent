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
    "左室侧后支 (L-PLB)", "右室侧后支 (R-PLB)"
]

# -----------------------------------------------------
# 2) Table columns (after "冠脉节段") and dropdown setup
# -----------------------------------------------------
all_columns = ["斑块种类", "类型", "症状", "大小(mm)", "狭窄程度", "闭塞"]

dropdown_columns = ["斑块种类", "狭窄程度", "闭塞"]  # use Combobox
blank_columns = ["类型", "症状", "大小(mm)"]        # use Entry

dropdown_options = {
    "斑块种类": ["NONE", "软斑块（非钙化性斑块）", "混合密度斑块", "硬斑块（钙化性斑块）"],
    "狭窄程度": ["NONE", "局限性狭窄", "阶段性狭窄", "弥漫性狭窄"],
    "闭塞": ["NONE", "是", "否"]
}

# ----------------------------------------------------------------
# 3) Top boxes: "冠状动脉钙化总积分", "LM", "LAD", "LCX", "RCA", etc.
#    and dropdown for "冠状动脉起源、走形及终止" + "冠脉优势型"
# ----------------------------------------------------------------
def show_popup_with_df(df: pd.DataFrame, top_data: dict):
    """
    Creates a popup window with:
      - Top fields (冠状动脉钙化总积分, LM, LAD, LCX, RCA as Entry;
        冠状动脉起源、走形及终止, 冠脉优势型 as Combobox)
      - Main table of segments vs. columns (some are dropdown, some are blank Entry)
    
    Args:
      df: DataFrame with columns [斑块种类, 类型, 症状, 大小(mm), 狭窄程度, 闭塞]
          and one row per segment.
      top_data: dict with keys:
        "冠状动脉钙化总积分", "LM", "LAD", "LCX", "RCA" (text)
        "冠状动脉起源、走形及终止" (dropdown: ["正常", "异常"])
        "冠脉优势型" (dropdown: ["右冠优势型", "左冠优势型", "均衡性"])
    """
    root = tk.Tk()
    root.title("冠脉结构化填写")
    
    # Configure the style for Combobox widgets to improve text visibility
    style = ttk.Style()
    style.map('TCombobox', fieldbackground=[('readonly', 'white')], 
              selectbackground=[('readonly', '#0078d7')], 
              selectforeground=[('readonly', 'white')])
    style.configure('TCombobox', background='white', foreground='black')

    # =========================
    # Top Frame (boxes/fields)
    # =========================
    top_frame = ttk.Frame(root)
    top_frame.pack(padx=10, pady=5, fill="x")

    # Row 0: 冠状动脉钙化总积分, LM, LAD, LCX, RCA (all text fields)
    ttk.Label(top_frame, text="冠状动脉钙化总积分:").grid(row=0, column=0, sticky="e", padx=5)
    entry_calc = ttk.Entry(top_frame, width=10)
    entry_calc.grid(row=0, column=1)
    entry_calc.insert(0, top_data.get("冠状动脉钙化总积分", "NONE"))

    ttk.Label(top_frame, text="LM:").grid(row=0, column=2, sticky="e", padx=5)
    entry_lm = ttk.Entry(top_frame, width=10)
    entry_lm.grid(row=0, column=3)
    entry_lm.insert(0, top_data.get("LM", "NONE"))

    ttk.Label(top_frame, text="LAD:").grid(row=0, column=4, sticky="e", padx=5)
    entry_lad = ttk.Entry(top_frame, width=10)
    entry_lad.grid(row=0, column=5)
    entry_lad.insert(0, top_data.get("LAD", "NONE"))

    ttk.Label(top_frame, text="LCX:").grid(row=0, column=6, sticky="e", padx=5)
    entry_lcx = ttk.Entry(top_frame, width=10)
    entry_lcx.grid(row=0, column=7)
    entry_lcx.insert(0, top_data.get("LCX", "NONE"))

    ttk.Label(top_frame, text="RCA:").grid(row=0, column=8, sticky="e", padx=5)
    entry_rca = ttk.Entry(top_frame, width=10)
    entry_rca.grid(row=0, column=9)
    entry_rca.insert(0, top_data.get("RCA", "NONE"))

    # Row 1: 冠状动脉起源、走形及终止 (dropdown), 冠脉优势型 (dropdown)
    ttk.Label(top_frame, text="冠状动脉起源、走形及终止:").grid(row=1, column=0, sticky="e", padx=5)
    combo_origin = ttk.Combobox(top_frame, values=["NONE", "正常", "异常"], width=8, state="readonly")
    combo_origin.grid(row=1, column=1)
    combo_origin.set(top_data.get("冠状动脉起源、走形及终止", "NONE"))

    ttk.Label(top_frame, text="冠脉优势型:").grid(row=1, column=2, sticky="e", padx=5)
    combo_dominance = ttk.Combobox(top_frame, values=["NONE", "右冠优势型", "左冠优势型", "均衡性"], width=10, state="readonly")
    combo_dominance.grid(row=1, column=3)
    combo_dominance.set(top_data.get("冠脉优势型", "NONE"))

    # =========================
    # Main Table (segments)
    # =========================
    table_frame = ttk.Frame(root)
    table_frame.pack(padx=10, pady=5)

    # Header row
    ttk.Label(table_frame, text="冠脉节段", borderwidth=1, relief="solid", width=20)\
        .grid(row=0, column=0)
    for j, col in enumerate(all_columns):
        ttk.Label(table_frame, text=col, borderwidth=1, relief="solid", width=15)\
            .grid(row=0, column=j+1)

    # Data rows
    for i, seg in enumerate(segments):
        ttk.Label(table_frame, text=seg, borderwidth=1, relief="solid", width=20)\
            .grid(row=i+1, column=0)

        for j, col in enumerate(all_columns):
            raw_val = df.at[i, col] if (col in df.columns and i in df.index) else None
            value = str(raw_val) if pd.notna(raw_val) else "NONE"

            if col in dropdown_columns:
                # Combobox with improved visibility
                widget = ttk.Combobox(table_frame, width=13, state="readonly")
                widget['values'] = dropdown_options[col]
                widget.grid(row=i+1, column=j+1)
                widget.set(value)
            else:
                # Blank text field
                widget = ttk.Entry(table_frame, width=15)
                widget.grid(row=i+1, column=j+1)
                widget.insert(0, value)

    root.mainloop()

# -----------------------------------------------------
# Test with dummy data (top fields + main table "NONE")
# -----------------------------------------------------
def test_gui_with_dummy_df():
    # 1) Build a dummy DataFrame for the table
    df_data = {"冠脉节段": segments}
    for col in all_columns:
        df_data[col] = ["NONE"] * len(segments)
    df = pd.DataFrame(df_data)
    df.reset_index(drop=True, inplace=True)

    # 2) Build a dict for the top fields
    top_data = {
        "冠状动脉钙化总积分": "NONE",
        "LM": "NONE",
        "LAD": "NONE",
        "LCX": "NONE",
        "RCA": "NONE",
        "冠状动脉起源、走形及终止": "NONE",
        "冠脉优势型": "NONE"
    }

    # 3) Show the popup
    show_popup_with_df(df, top_data)

if __name__ == "__main__":
    test_gui_with_dummy_df()
