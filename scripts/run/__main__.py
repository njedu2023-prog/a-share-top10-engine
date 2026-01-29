from __future__ import annotations

import json
import os
import re
import sys
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz
import yaml
from dateutil import tz
from loguru import logger


# -----------------------------
# Config models (lightweight)
# -----------------------------
@dataclass
class DataRepo:
    owner: str
    name: str
    ref: str = "main"
    checkout_dir: str = "_warehouse"


@dataclass
class Settings:
    timezone: str
    encoding: str
    top_n: int
    data_repo: DataRepo
    trade_date_dirs: List[str]
    snapshot_candidates: Dict[str, List[str]]
    output_files: Dict[str, str]
    weights: Dict[str, float]
    prob_map: Dict[str, Any]
    learning_storage_file: str


# -----------------------------
# Utils
# -----------------------------
def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_str(tz_name: str) -> str:
    z = pytz.timezone(tz_name)
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())) if z is None else pd.Timestamp.now(tz=z).strftime("%Y%m%d_%H%M%S")


def _git_clone_or_update(owner: str, repo: str, ref: str, dest: Path) -> None:
    """
    Actions 环境里通常有 git。这里用 https clone（公共仓库无需 token）。
    若目录已存在，就尝试 fetch + reset 到指定 ref。
    """
    url = f"https://github.com/{owner}/{repo}.git"
    if not dest.exists():
        logger.info(f"克隆数据仓库: {url} -> {dest} (ref={ref})")
        _ensure_dir(dest.parent)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ref, url, str(dest)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return

    # 已存在：尽量更新
    logger.info(f"更新数据仓库: {dest} (ref={ref})")
    subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", ref], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", ref], check=True)
    subprocess.run(["git", "-C", str(dest), "reset", "--hard", f"origin/{ref}"], check=True)


def _extract_trade_date_candidates(path: Path) -> List[str]:
    """
    从目录名/文件名中抽取 YYYYMMDD
    """
    pat = re.compile(r"(20\d{2}[01]\d[0-3]\d)")
    found: List[str] = []
    if not path.exists():
        return found

    # 只扫一层，避免巨大 IO
    for child in path.iterdir():
        m = pat.search(child.name)
        if m:
            found.append(m.group(1))
    return found


def _resolve_trade_date_dir(settings: Settings) -> Tuple[str, Path]:
    """
    ✅工程级交易日解析（最终稳态版）

    - 支持 TRADE_DATE 指定目标日
    - 目标日不存在：fallback 到仓库内最新可用交易日
    - 未指定：直接用最新可用交易日
    - 扫描逻辑对模板更鲁棒，不依赖当前工作目录
    """

    # 0) 读取目标日
    target_date = os.getenv("TRADE_DATE", "").strip()
    if target_date:
        logger.info(f"收到用户指定 TRADE_DATE={target_date}")

    # 1) 统一把相对路径锚定到仓库根目录（从 checkout_dir 推断）
    # settings.trade_date_dirs 通常是 _warehouse/... 开头
    repo_root = Path(__file__).resolve().parents[2]
    warehouse_root = (repo_root / settings.data_repo.checkout_dir).resolve()

    # 2) 构造“扫描根目录”集合：从 trade_date_dirs 推断出若干可扫描父目录
    scan_roots: List[Path] = []
    pat = re.compile(r"(20\d{2}[01]\d[0-3]\d)")

    for tmpl in settings.trade_date_dirs:
        s = tmpl

        # 把模板路径锚定到 repo_root
        p = (repo_root / s).resolve() if not Path(s).is_absolute() else Path(s).resolve()

        # 找到模板里 {trade_date} 的位置，取它前面的父目录作为扫描根
        if "{trade_date}" in s:
            # 取 "{trade_date}" 之前的路径部分
            prefix = s.split("{trade_date}", 1)[0].rstrip("/")

            # prefix 可能是 "_warehouse/data/"，我们锚定到 repo_root
            root = (repo_root / prefix).resolve() if not Path(prefix).is_absolute() else Path(prefix).resolve()
            scan_roots.append(root)
        else:
            # 没有占位符：就用它自身的父目录去扫
            scan_roots.append(p.parent)

    # 去重 & 过滤不存在的
    uniq_roots = []
    for r in scan_roots:
        if r.exists() and r not in uniq_roots:
            uniq_roots.append(r)

    # 3) 扫描所有可用日期
    all_dates: List[Tuple[str, Path]] = []

    for root in uniq_roots:
        # 只扫一层（你原来的策略）
        for child in root.iterdir():
            m = pat.search(child.name)
            if not m:
                continue
            d = m.group(1)

            # 用所有模板去验证真实存在的 trade_dir
            for tmpl in settings.trade_date_dirs:
                dpath = (repo_root / tmpl.format(trade_date=d)).resolve() if not Path(tmpl).is_absolute() else Path(tmpl.format(trade_date=d)).resolve()
                if dpath.exists():
                    all_dates.append((d, dpath))

    if not all_dates:
        tried = "\n".join([f"- {x}" for x in settings.trade_date_dirs])
        raise FileNotFoundError(
            "无法定位任何交易日目录（仓库里完全没有 YYYYMMDD 目录）。\n"
            "已尝试路径模板:\n"
            f"{tried}\n"
            f"提示：请检查 {warehouse_root} 下是否存在 data/20260129 这种结构。"
        )

    # 取最新
    all_dates.sort(key=lambda x: x[0])
    latest_date, latest_dir = all_dates[-1]

    # 4) 若指定目标日：优先尝试
    if target_date:
        for tmpl in settings.trade_date_dirs:
            dpath = (repo_root / tmpl.format(trade_date=target_date)).resolve() if not Path(tmpl).is_absolute() else Path(tmpl.format(trade_date=target_date)).resolve()
            if dpath.exists():
                logger.info(f"✅目标交易日目录存在: {target_date} -> {dpath}")
                return target_date, dpath

        logger.warning(f"⚠️目标交易日 {target_date} 不存在，自动回退到最近可用交易日 {latest_date}")
        return latest_date, latest_dir

    # 5) 未指定：直接用最新
    logger.info(f"未指定 trade_date，自动选择最新可用交易日: {latest_date}")
    return latest_date, latest_dir



def _find_snapshot_file(trade_dir: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = (trade_dir / name).resolve()
        if p.exists() and p.is_file():
            return p
    return None


def _sigmoid(x: float, k: float, x0: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def _normalize_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(float)
    mn, mx = s2.min(), s2.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([0.5] * len(s2), index=s2.index)
    return (s2 - mn) / (mx - mn)


# -----------------------------
# Core pipeline (MVP runnable)
# -----------------------------
def build_settings(cfg: Dict[str, Any]) -> Settings:
    runtime = cfg.get("runtime", {})
    project = cfg.get("project", {})
    data_repo = cfg.get("data_repo", {})
    layout = cfg.get("data_layout", {})
    output = cfg.get("output", {})
    scoring = cfg.get("scoring", {})
    learning = cfg.get("learning", {})

    timezone = runtime.get("timezone", "Asia/Shanghai")
    encoding = runtime.get("encoding", "utf-8")
    top_n = int(project.get("top_n", 10))

    dr = DataRepo(
        owner=str(data_repo.get("owner", "")),
        name=str(data_repo.get("name", "")),
        ref=str(data_repo.get("ref", "main")),
        checkout_dir=str(data_repo.get("checkout_dir", "_warehouse")),
    )
    if not dr.owner or not dr.name:
        raise ValueError("config/settings.yaml 缺少 data_repo.owner 或 data_repo.name")

    trade_date_dirs = list(layout.get("trade_date_dirs", []))
    snapshot_candidates = dict(layout.get("snapshot_candidates", {}))

    files = (output.get("files", {}) or {})
    # 必须要输出这两个
    if "predict_json" not in files or "predict_md" not in files:
        raise ValueError("output.files 至少需要包含 predict_json / predict_md")

    weights = (scoring.get("weights", {}) or {})
    prob_map = (scoring.get("prob_map", {}) or {"method": "sigmoid", "k": 6.0, "x0": 0.5})

    storage = (learning.get("storage", {}) or {})
    storage_file = str(storage.get("file", "outputs/verify/history.jsonl"))

    return Settings(
        timezone=timezone,
        encoding=encoding,
        top_n=top_n,
        data_repo=dr,
        trade_date_dirs=trade_date_dirs,
        snapshot_candidates=snapshot_candidates,
        output_files=files,
        weights=weights,
        prob_map=prob_map,
        learning_storage_file=storage_file,
    )


def load_limitup_pool(trade_dir: Path, candidates: Dict[str, List[str]]) -> pd.DataFrame:
    key = "limit_up_pool"
    if key not in candidates:
        raise ValueError("snapshot_candidates 缺少 limit_up_pool")
    f = _find_snapshot_file(trade_dir, candidates[key])
    if f is None:
        raise FileNotFoundError(f"未找到涨停池文件。已尝试: {candidates[key]}，目录: {trade_dir}")
    df = pd.read_csv(f)

    # 宽松字段适配：code / ts_code / 股票代码
    possible_code_cols = ["ts_code", "code", "symbol", "股票代码", "证券代码"]
    code_col = next((c for c in possible_code_cols if c in df.columns), None)
    if code_col is None:
        raise ValueError(f"涨停池文件缺少代码列（ts_code/code/...）。当前列: {list(df.columns)}")

    df = df.rename(columns={code_col: "code"}).copy()
    df["code"] = df["code"].astype(str)

    # 尝试找“连板数/封单/强度”等字段（没有也能跑）
    # 常见：lb/连板/board/连板数，fd/封单/封单额，pct/涨幅
    return df


def score_stocks(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    MVP：在没有完整五因子数据时，做一个“可运行+可扩展”的打分框架：
    - 如果存在某些字段，就参与打分；不存在则给中性值 0.5
    """
    df = df.copy()

    # 1) 连板动量
    col_momentum = next((c for c in ["连板", "连板数", "board", "lb", "streak"] if c in df.columns), None)
    momentum = _normalize_series(df[col_momentum]) if col_momentum else pd.Series([0.5] * len(df))

    # 2) 封单强度
    col_imb = next((c for c in ["封单额", "封单", "order_imbalance", "imbalance", "fd"] if c in df.columns), None)
    imbalance = _normalize_series(df[col_imb]) if col_imb else pd.Series([0.5] * len(df))

    # 3) 热点概念（如果有概念热度字段就用）
    col_concept = next((c for c in ["概念热度", "concept_hot", "hot_concept", "theme_hot"] if c in df.columns), None)
    hot_concept = _normalize_series(df[col_concept]) if col_concept else pd.Series([0.5] * len(df))

    # 4) 龙虎榜（如果有龙虎榜标记/强度字段就用）
    col_lhb = next((c for c in ["龙虎榜", "dragon_tiger", "lhb", "dt_score"] if c in df.columns), None)
    dragon_tiger = _normalize_series(df[col_lhb]) if col_lhb else pd.Series([0.5] * len(df))

    # 5) 大盘过滤（若没有则中性）
    col_market = next((c for c in ["market_filter", "指数强度", "market_score"] if c in df.columns), None)
    market_filter = _normalize_series(df[col_market]) if col_market else pd.Series([0.5] * len(df))

    w1 = float(weights.get("w_limitup_momentum", 0.30))
    w2 = float(weights.get("w_order_imbalance", 0.20))
    w3 = float(weights.get("w_hot_concept", 0.20))
    w4 = float(weights.get("w_dragon_tiger", 0.15))
    w5 = float(weights.get("w_market_filter", 0.15))

    df["score"] = (
        w1 * momentum.values
        + w2 * imbalance.values
        + w3 * hot_concept.values
        + w4 * dragon_tiger.values
        + w5 * market_filter.values
    )

    return df


def to_probability(df: pd.DataFrame, prob_map: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    method = str(prob_map.get("method", "sigmoid")).lower()
    if method != "sigmoid":
        # 默认回退
        method = "sigmoid"
    k = float(prob_map.get("k", 6.0))
    x0 = float(prob_map.get("x0", 0.5))
    df["prob"] = df["score"].apply(lambda x: float(_sigmoid(float(x), k=k, x0=x0)))
    return df


def write_outputs(trade_date: str, top_df: pd.DataFrame, settings: Settings) -> Tuple[Path, Path]:
    predict_json = Path(settings.output_files["predict_json"].format(trade_date=trade_date))
    predict_md = Path(settings.output_files["predict_md"].format(trade_date=trade_date))

    _ensure_dir(predict_json.parent)
    _ensure_dir(predict_md.parent)

    records = []
    for i, row in top_df.reset_index(drop=True).iterrows():
        records.append(
            {
                "rank": int(i + 1),
                "code": str(row.get("code", "")),
                "score": float(row.get("score", 0.0)),
                "prob": float(row.get("prob", 0.0)),
            }
        )

    payload = {
        "trade_date": trade_date,
        "generated_at": pd.Timestamp.now(tz=pytz.timezone(settings.timezone)).isoformat(),
        "top_n": settings.top_n,
        "items": records,
    }

    predict_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding=settings.encoding)

    lines = []
    lines.append(f"# Top{settings.top_n} 预测结果（trade_date={trade_date}）")
    lines.append("")
    lines.append("| Rank | Code | Score | Prob |")
    lines.append("|---:|---|---:|---:|")
    for r in records:
        lines.append(f"| {r['rank']} | {r['code']} | {r['score']:.4f} | {r['prob']:.4f} |")
    lines.append("")
    lines.append("> 说明：当前为工程可跑 MVP 打分框架；后续你提供 PDF 的严格因子/权重后，我再帮你对齐。")

    predict_md.write_text("\n".join(lines), encoding=settings.encoding)

    return predict_json, predict_md


def append_history(trade_date: str, top_df: pd.DataFrame, settings: Settings) -> None:
    """
    把本次预测写入 history.jsonl（用于后续回测/命中率统计）
    """
    hist_path = Path(settings.learning_storage_file)
    _ensure_dir(hist_path.parent)

    ts = pd.Timestamp.now(tz=pytz.timezone(settings.timezone)).isoformat()
    for i, row in top_df.reset_index(drop=True).iterrows():
        obj = {
            "ts": ts,
            "trade_date": trade_date,
            "rank": int(i + 1),
            "code": str(row.get("code", "")),
            "score": float(row.get("score", 0.0)),
            "prob": float(row.get("prob", 0.0)),
        }
        with hist_path.open("a", encoding=settings.encoding) as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]  # scripts/run/__main__.py -> repo
    cfg_path = repo_root / "config" / "settings.yaml"

    # logs
    cfg_raw = _read_yaml(cfg_path)
    settings = build_settings(cfg_raw)

    logs_dir = repo_root / "outputs" / "logs"
    _ensure_dir(logs_dir)
    log_file = logs_dir / f"run_{_now_str(settings.timezone)}.log"
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(str(log_file), level="INFO", encoding=settings.encoding)

    logger.info(f"Repo root: {repo_root}")
    logger.info(f"Config: {cfg_path}")

    # 1) 准备数据仓库（clone/update）
    dr = settings.data_repo
    warehouse = (repo_root / dr.checkout_dir).resolve()
    _git_clone_or_update(dr.owner, dr.name, dr.ref, warehouse)

    # 2) 在数据仓库里定位最新交易日目录
    # 注意：settings.trade_date_dirs 里默认以 _warehouse/... 开头，因此这里不需要额外拼接
    trade_date, trade_dir = _resolve_trade_date_dir(settings)
    logger.info(f"定位到交易日: {trade_date}，目录: {trade_dir}")

    # 3) 读取涨停池并打分
    df_pool = load_limitup_pool(trade_dir, settings.snapshot_candidates)
    logger.info(f"涨停池载入成功: rows={len(df_pool)}, cols={len(df_pool.columns)}")

    df_scored = score_stocks(df_pool, settings.weights)
    df_prob = to_probability(df_scored, settings.prob_map)

    # 4) 输出 TopN
    top_df = df_prob.sort_values(["score"], ascending=False).head(settings.top_n).reset_index(drop=True)
    predict_json, predict_md = write_outputs(trade_date, top_df, settings)
    logger.info(f"输出完成: {predict_json}")
    logger.info(f"输出完成: {predict_md}")

    # 5) 追加历史
    append_history(trade_date, top_df, settings)
    logger.info(f"历史沉淀完成: {settings.learning_storage_file}")

    logger.info("运行成功 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
