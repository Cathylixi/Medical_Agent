import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
from rapidfuzz import fuzz, process

KB_PATH = Path("data/medical_terms.json")


def _load_kb() -> List[Dict[str, Any]]:
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found: {KB_PATH}")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_alias_index(kb: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Return alias->canonical and canonical->meta mapping.
    All keys in alias map are lower-cased.
    """
    alias_to_canonical: Dict[str, str] = {}
    canonical_to_meta: Dict[str, Dict[str, Any]] = {}

    for item in kb:
        cn = (item.get("中文名称") or "").strip()
        abbr = (item.get("测量值简写") or "").strip()
        aliases = item.get("别名", []) or []
        if isinstance(aliases, str):
            aliases = [a.strip() for a in aliases.split(";") if a.strip()]
        names = set([cn, abbr]) | set(aliases)
        for n in names:
            if n:
                alias_to_canonical[n.lower()] = cn
        canonical_to_meta[cn] = item
    return alias_to_canonical, canonical_to_meta


def _match_name(name: str, alias_to_canonical: Dict[str, str]) -> str:
    """Return canonical name using exact or fuzzy match, else empty string."""
    if not name:
        return ""
    key = name.strip().lower()
    if key in alias_to_canonical:
        return alias_to_canonical[key]
    # fuzzy: try stripping parentheses content
    base = name.split("(")[0].strip().lower()
    if base in alias_to_canonical:
        return alias_to_canonical[base]
    # rapidfuzz fallback
    keys = list(alias_to_canonical.keys())
    if not keys:
        return ""
    best = process.extractOne(key, keys, scorer=fuzz.WRatio)
    if best and best[1] >= 88:
        return alias_to_canonical[best[0]]
    best = process.extractOne(base, keys, scorer=fuzz.WRatio)
    if best and best[1] >= 88:
        return alias_to_canonical[best[0]]
    return ""


def normalize_table_with_kb(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize formatted table using medical_terms.json.

    - Standardize 名称 to "中文名称(测量值简写)" if possible
    - Fill 英文 with 测量值英文 when empty
    - Fill 单位 when empty
    
    This function returns a new DataFrame instance (does not mutate input).
    """
    if df is None or df.empty:
        return df

    try:
        kb = _load_kb()
    except FileNotFoundError:
        # no KB, skip
        return df

    alias_to_canonical, canonical_meta = _build_alias_index(kb)

    df = df.copy()
    for i in range(len(df)):
        name = str(df.at[i, "名称"]) if "名称" in df.columns else ""
        canonical = _match_name(name, alias_to_canonical)
        if not canonical:
            # try split
            base = name.split("(")[0].strip()
            canonical = _match_name(base, alias_to_canonical)
        if not canonical:
            continue

        meta = canonical_meta.get(canonical, {})
        abbr = (meta.get("测量值简写") or "").strip()
        english = (meta.get("测量值英文") or "").strip()
        unit = (meta.get("单位") or "").strip()

        # 标准化名称为 中文名称(简写) 若有简写
        if abbr:
            std_name = f"{canonical}({abbr})"
        else:
            std_name = canonical
        df.at[i, "名称"] = std_name

        # 填英文
        if "英文" in df.columns:
            if not isinstance(df.at[i, "英文"], str) or not df.at[i, "英文"].strip():
                if english:
                    df.at[i, "英文"] = english
        # 填单位（若为空）
        if "单位" in df.columns:
            cur_unit = str(df.at[i, "单位"]) if df.at[i, "单位"] is not None else ""
            if not cur_unit.strip() and unit:
                df.at[i, "单位"] = unit

    return df 