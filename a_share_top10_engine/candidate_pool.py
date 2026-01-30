# a_share_top10_engine/candidate_pool.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import pandas as pd

from .data_loader import DataLoader, SnapshotPaths


# ============================
# Column Helpers (robust)
# ============================

def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """从候选列名中挑第一个存在的列名。"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_code_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    统一股票代码列为 ts_code（返回新df和实际使用的代码列名）。
    """
    code_col = _pick_first_existing(
        df,
        ["ts_code", "code", "stock_code", "symbol", "sec_code", "tscode"],
    )
    if code_col is None:
        raise KeyError(
            f"[candidate_pool] 无法识别股票代码列。已有列：{list(df.columns)}"
        )
    if code_col != "ts_code":
        df = df.rename(columns={code_col: "ts_code"})
    # 统一成字符串，避免 merge 失败
    df["ts_code"] = df["ts_code"].astype(str)
    return df, "ts_code"


def _safe_numeric(s: pd.Series) -> pd.Series:
    """把可能混杂字符串的列尽量转成数值。"""
    return pd.to_numeric(s, errors="coerce")


def _derive_break_rate(limit_break: pd.DataFrame) -> pd.DataFrame:
    """
    从炸板表推导 break_rate（尽可能适配不同字段结构）。
    输出两列：ts_code, break_rate
    """
    df = limit_break.copy()
    df, _ = _ensure_code_column(df)

    # 1) 直接已有 break_rate
    br_col = _pick_first_existing(df, ["break_rate", "zb_rate", "炸板率"])
    if br_col:
        out = df[["ts_code", br_col]].copy()
        out = out.rename(columns={br_col: "break_rate"})
        out["break_rate"] = _safe_numeric(out["break_rate"])
        return out.drop_duplicates("ts_code")

    # 2) 用 break_times / total_times 推
    break_times_col = _pick_first_existing(df, ["break_times", "zb_times", "炸板次数", "break_cnt"])
    total_times_col = _pick_first_existing(df, ["total_times", "total_cnt", "尝试次数", "hit_times", "limit_times"])

    if break_times_col and total_times_col:
        out = df[["ts_code", break_times_col, total_times_col]].copy()
        out[break_times_col] = _safe_numeric(out[break_times_col])
        out[total_times_col] = _safe_numeric(out[total_times_col]).replace(0, pd.NA)
        out["break_rate"] = out[break_times_col] / out[total_times_col]
        return out[["ts_code", "break_rate"]].drop_duplicates("ts_code")

    # 3) 推不出来就给空列（不报错，交给上层处理）
    out = df[["ts_code"]].copy()
    out["break_rate"] = pd.NA
    return out.drop_duplicates("ts_code")


def _extract_amount(daily: pd.DataFrame) -> pd.DataFrame:
    """
    从日线表中抽取成交额 amount（尽量适配不同列名）。
    输出：ts_code, amount
    """
    df = daily.copy()
    df, _ = _ensure_code_column(df)

    amt_col = _pick_first_existing(df, ["amount", "amt", "turnover", "成交额"])
    if amt_col is None:
        out = df[["ts_code"]].copy()
        out["amount"] = pd.NA
        return out.drop_duplicates("ts_code")

    out = df[["ts_code", amt_col]].copy().rename(columns={amt_col: "amount"})
    out["amount"] = _safe_numeric(out["amount"])
    return out.drop_duplicates("ts_code")


def _extract_core_tag(tags: pd.DataFrame) -> pd.DataFrame:
    """
    从题材标签表抽取“是否核心题材”字段 tag_core（尽量适配不同列名）。
    输出：ts_code, tag_core, (可选) tag_name
    """
    df = tags.copy()
    df, _ = _ensure_code_column(df)

    core_col = _pick_first_existing(
        df,
        ["tag_core", "is_core", "core", "is_hot", "is_main", "核心", "核心题材"],
    )
    name_col = _pick_first_existing(df, ["tag", "tag_name", "concept", "theme", "题材", "题材名称"])

    cols = ["ts_code"]
    if core_col:
        cols.append(core_col)
    if name_col:
        cols.append(name_col)

    out = df[cols].copy()
    if core_col:
        out = out.rename(columns={core_col: "tag_core"})
    else:
        out["tag_core"] = pd.NA

    if name_col:
        out = out.rename(columns={name_col: "tag_name"})
    else:
        out["tag_name"] = pd.NA

    # 规范 tag_core 为 0/1/NA（尽量）
    if "tag_core" in out.columns:
        # 可能是 True/False, "Y/N", "核心/非核心", 1/0 等
        out["tag_core"] = out["tag_core"].map(
            lambda x: 1
            if str(x).strip().lower() in ["1", "true", "t", "y", "yes", "核心", "core"]
            else (0 if str(x).strip().lower() in ["0", "false", "f", "n", "no", "非核心", "not core"] else pd.NA)
        )

    return out.drop_duplicates("ts_code")


# ============================
# Candidate Pool Builder
# ============================

@dataclass
class CandidatePoolCfg:
    """
    候选池构建配置（只做轻量过滤，不做评分/概率）。
    """
    min_amount: Optional[float] = None  # 成交额下限（如 5e8），None 表示不筛
    require_tag_core: bool = False      # 是否只保留核心题材
    dropna_ts_code: bool = True         # 是否剔除无代码行


def build_candidates(
    trade_date: str,
    paths: SnapshotPaths | None = None,
    cfg: CandidatePoolCfg | None = None,
    keep_raw: bool = False,
) -> pd.DataFrame:
    """
    构建“首板候选池”（模块②第一步闭环）

    输入：
      - trade_date: 'YYYYMMDD'
      - paths: SnapshotPaths（可选，默认仓库约定）
      - cfg: 候选池过滤配置（可选）
      - keep_raw: 是否保留原始表的更多字段（默认False，仅输出核心字段）

    输出：DataFrame（至少包含）
      - ts_code
      - break_rate
      - amount
      - tag_core
      - tag_name（若有）
    """
    cfg = cfg or CandidatePoolCfg()

    loader = DataLoader(trade_date=trade_date, paths=paths)

    limit_list = loader.load_limit_list()
    limit_break = loader.load_limit_break()
    daily = loader.load_daily()
    tags = loader.load_limit_up_tags()

    # 统一代码列
    limit_list, _ = _ensure_code_column(limit_list)

    # 核心字段抽取
    br = _derive_break_rate(limit_break)
    amt = _extract_amount(daily)
    tg = _extract_core_tag(tags)

    # 以涨停池为基底（候选入口）
    base_cols = ["ts_code"]
    base = limit_list.copy()
    if cfg.dropna_ts_code:
        base = base.dropna(subset=["ts_code"])
    base["ts_code"] = base["ts_code"].astype(str)

    if not keep_raw:
        base = base[base_cols].drop_duplicates("ts_code")
    else:
        base = base.drop_duplicates("ts_code")

    # 逐步 merge（左连接，保证候选池来自 limit_list）
    out = base.merge(br, on="ts_code", how="left")
    out = out.merge(amt, on="ts_code", how="left")
    out = out.merge(tg, on="ts_code", how="left")

    # 过滤（只做轻量过滤）
    if cfg.min_amount is not None:
        out = out[_safe_numeric(out["amount"]) >= cfg.min_amount]

    if cfg.require_tag_core:
        out = out[out["tag_core"] == 1]

    # 输出字段排序（工程可读）
    preferred_order = ["ts_code", "amount", "break_rate", "tag_core", "tag_name"]
    cols = [c for c in preferred_order if c in out.columns] + [c for c in out.columns if c not in preferred_order]
    out = out[cols].reset_index(drop=True)

    return out


# ============================
# Quick Test入口（本地调试用）
# ============================

if __name__ == "__main__":
    # 示例：仅构建候选池，不做过滤
    trade_date = "20260129"
    df = build_candidates(trade_date=trade_date)

    print("\n======================")
    print("Candidate Pool Built")
    print("trade_date:", trade_date)
    print("shape:", df.shape)
    print(df.head(20))
