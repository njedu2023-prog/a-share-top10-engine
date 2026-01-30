from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from loguru import logger


# -------------------------
# Models
# -------------------------

@dataclass
class DataRepo:
    owner: str
    name: str
    ref: str = "main"
    checkout_dir: str = "_warehouse"

@dataclass
class ProbMap:
    method: str = "sigmoid"
    k: float = 0.18

@dataclass
class OutputCfg:
    dir: str = "outputs"
    files: Dict[str, str] = None  # predict_json / predict_md
    history_append_jsonl: str = "outputs/history/history.jsonl"

@dataclass
class Settings:
    encoding: str = "utf-8"
    timezone: str = "Asia/Shanghai"
    top_n: int = 10

    data_repo: DataRepo = None
    trade_date_dirs: List[str] = None
    snapshot_candidates: List[str] = None

    weights: Dict[str, float] = None
    prob_map: ProbMap = ProbMap()

    output: OutputCfg = None


# -------------------------
# Utils
# -------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _now_str(tz: str) -> str:
    # GitHub runner 不一定有 tz database 完整配置；这里用本地时间即可
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _read_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    logger.info(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def _git_clone_or_update(owner: str, name: str, ref: str, dest: Path) -> None:
    url = f"https://github.com/{owner}/{name}.git"
    if dest.exists() and (dest / ".git").exists():
        logger.info(f"更新数据仓库: {url} -> {dest} (ref={ref})")
        _run(["git", "fetch", "--all", "--prune"], cwd=dest)
        _run(["git", "checkout", ref], cwd=dest)
        _run(["git", "pull", "--ff-only"], cwd=dest)
    else:
        logger.info(f"克隆数据仓库: {url} -> {dest} (ref={ref})")
        _ensure_dir(dest.parent)
        _run(["git", "clone", "--depth", "1", "--branch", ref, url, str(dest)])

def _fmt_template(tpl: str, yyyymmdd: str) -> str:
    yyyy = yyyymmdd[:4]
    return (
        tpl.replace("{YYYYMMDD}", yyyymmdd)
           .replace("{YYYY}", yyyy)
    )

def _extract_yyyymmdd_from_path(p: Path) -> Optional[str]:
    # 抓取路径里最像 YYYYMMDD 的段
    m = re.search(r"(20\d{2}[01]\d[0-3]\d)", str(p))
    return m.group(1) if m else None


# -------------------------
# Build settings (fixes your errors)
# -------------------------

def build_settings(cfg: Dict[str, Any]) -> Settings:
    # 1) data_repo 必填修复
    dr = cfg.get("data_repo") or {}
    if not dr.get("owner") or not dr.get("name"):
        raise ValueError("config/settings.yaml 缺少 data_repo.owner 或 data_repo.name")

    # 2) output.files 必须含 predict_json/predict_md 修复
    out = cfg.get("output") or {}
    out_files = out.get("files") or {}
    if "predict_json" not in out_files or "predict_md" not in out_files:
        raise ValueError("output.files 至少需要包含 predict_json / predict_md")

    prob_raw = cfg.get("prob_map") or {}
    prob = ProbMap(
        method=str(prob_raw.get("method", "sigmoid")),
        k=float(prob_raw.get("k", 0.18)),
    )

    s = Settings(
        encoding=str(cfg.get("encoding", "utf-8")),
        timezone=str(cfg.get("timezone", "Asia/Shanghai")),
        top_n=int(cfg.get("top_n", 10)),

        data_repo=DataRepo(
            owner=str(dr["owner"]),
            name=str(dr["name"]),
            ref=str(dr.get("ref", "main")),
            checkout_dir=str(dr.get("checkout_dir", "_warehouse")),
        ),

        trade_date_dirs=list(cfg.get("trade_date_dirs") or []),
        snapshot_candidates=list(cfg.get("snapshot_candidates") or []),

        weights=dict(cfg.get("weights") or {}),
        prob_map=prob,

        output=OutputCfg(
            dir=str(out.get("dir", "outputs")),
            files=dict(out_files),
            history_append_jsonl=str(out.get("history_append_jsonl", "outputs/history/history.jsonl")),
        ),
    )
    if not s.trade_date_dirs:
        raise ValueError("trade_date_dirs 不能为空（用于定位交易日目录）")
    if not s.snapshot_candidates:
        raise ValueError("snapshot_candidates 不能为空（用于定位涨停池快照文件）")
    return s


# -------------------------
# Trade date dir resolve (fixes your errors)
# -------------------------

def _resolve_trade_date_dir(settings: Settings) -> Tuple[str, Path]:
    # 优先用用户指定
    user_td = os.getenv("TRADE_DATE", "").strip()
    if user_td:
        logger.info(f"收到用户指定 TRADE_DATE={user_td}")
        trade_date = user_td
        candidates = [_fmt_template(tpl, trade_date) for tpl in settings.trade_date_dirs]
        for c in candidates:
            p = (Path(settings.output.dir).parents[0] / c).resolve()  # safe join-ish
            # 上面 join 对相对路径可能不准，这里更直接：以 repo_root 为基准在 main() 里处理
        # 在 main() 里我们会传 repo_root 进行 resolve，这里只返回日期，目录在 main 里二次找
        # 但为简洁，这里直接按 repo_root=当前工作目录处理：
        repo_root = Path.cwd().resolve()
        for c in candidates:
            p = (repo_root / c).resolve()
            if p.exists() and p.is_dir():
                return trade_date, p
        raise FileNotFoundError(
            "无法定位指定交易日目录。已尝试路径模板:\n" + "\n".join(candidates)
        )

    # 否则：自动在 _warehouse 下搜集所有 YYYYMMDD 目录并取最大
    repo_root = Path.cwd().resolve()
    warehouse = (repo_root / settings.data_repo.checkout_dir).resolve()
    if not warehouse.exists():
        raise FileNotFoundError(f"warehouse 不存在: {warehouse}")

    # 只在仓库里找形如 .../YYYY/YYYYMMDD 或 .../YYYYMMDD 的目录
    ymd_dirs: List[Path] = []
    for p in warehouse.rglob("*"):
        if p.is_dir():
            ymd = _extract_yyyymmdd_from_path(p)
            if ymd and Path(str(ymd)) == Path(ymd):  # trivial
                # 进一步要求目录名里包含 ymd（避免误判）
                if p.name == ymd:
                    ymd_dirs.append(p)

    if not ymd_dirs:
        raise FileNotFoundError(
            "无法定位任何交易日目录（仓库里完全没有 YYYYMMDD 目录）。\n"
            f"提示：请检查 {warehouse} 下是否存在 data/raw/2026/20260129 这种结构。"
        )

    ymd_dirs.sort(key=lambda x: x.name)
    trade_dir = ymd_dirs[-1]
    trade_date = trade_dir.name
    return trade_date, trade_dir


# -------------------------
# Load / Score / Output
# -------------------------

def load_limitup_pool(trade_dir: Path, snapshot_candidates: List[str]) -> pd.DataFrame:
    for fn in snapshot_candidates:
        fp = trade_dir / fn
        if fp.exists():
            df = pd.read_csv(fp)
            df["__source_file__"] = fn
            return df
    raise FileNotFoundError(
        "未找到任何涨停池候选文件: " + ", ".join(snapshot_candidates) + f" in {trade_dir}"
    )

def _to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def score_stocks(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    df2 = df.copy()

    # 常见字段兼容（不同数据快照字段名不一）
    rename_map = {
        "ts_code": "ts_code",
        "code": "ts_code",
        "symbol": "ts_code",
        "name": "name",
        "stock_name": "name",

        "pct_chg": "pct_chg",
        "pctchg": "pct_chg",

        "turnover_rate": "turnover_rate",
        "turnover": "turnover_rate",

        "amount": "amount",
        "成交额": "amount",

        "open_times": "open_times",
        "close_times": "close_times",
        "first_time": "first_time",
    }
    for k, v in rename_map.items():
        if k in df2.columns and v not in df2.columns:
            df2[v] = df2[k]

    # 计算 score：存在的字段才参与
    score = pd.Series(0.0, index=df2.index)
    for col, w in (weights or {}).items():
        if col in df2.columns:
            x = _to_numeric_safe(df2[col]).fillna(0.0)
            # 归一化（防止量纲差异过大）
            denom = (x.abs().mean() + 1e-9)
            score = score + (x / denom) * float(w)

    df2["score"] = score
    return df2

def to_probability(df: pd.DataFrame, prob_map: ProbMap) -> pd.DataFrame:
    df2 = df.copy()
    if prob_map.method.lower() == "sigmoid":
        k = float(prob_map.k)
        df2["prob"] = df2["score"].apply(lambda z: 1.0 / (1.0 + math.exp(-k * float(z))))
    else:
        # fallback: min-max
        mn, mx = df2["score"].min(), df2["score"].max()
        if mx - mn < 1e-9:
            df2["prob"] = 0.5
        else:
            df2["prob"] = (df2["score"] - mn) / (mx - mn)
    return df2

def write_outputs(trade_date: str, top_df: pd.DataFrame, settings: Settings) -> Tuple[Path, Path]:
    repo_root = Path.cwd().resolve()

    def render(path_tpl: str) -> Path:
        rel = _fmt_template(path_tpl, trade_date)
        p = (repo_root / rel).resolve()
        _ensure_dir(p.parent)
        return p

    predict_json = render(settings.output.files["predict_json"])
    predict_md = render(settings.output.files["predict_md"])

    # JSON
    payload = {
        "trade_date": trade_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "top_n": int(settings.top_n),
        "items": top_df.to_dict(orient="records"),
    }
    predict_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding=settings.encoding)

    # Markdown
    lines = []
    lines.append(f"# Top{settings.top_n} Prediction ({trade_date})")
    lines.append("")
    cols = [c for c in ["ts_code", "name", "score", "prob", "__source_file__"] if c in top_df.columns]
    if not cols:
        cols = list(top_df.columns[:5])
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in top_df.iterrows():
        row = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                row.append(f"{v:.6f}")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")
    predict_md.write_text("\n".join(lines) + "\n", encoding=settings.encoding)

    return predict_json, predict_md

def append_history(trade_date: str, top_df: pd.DataFrame, settings: Settings) -> None:
    repo_root = Path.cwd().resolve()
    hist_path = (repo_root / settings.output.history_append_jsonl).resolve()
    _ensure_dir(hist_path.parent)

    obj = {
        "trade_date": trade_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "items": top_df.to_dict(orient="records"),
    }
    with hist_path.open("a", encoding=settings.encoding) as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -------------------------
# Main
# -------------------------

def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]  # scripts/run/__main__.py -> repo
    os.chdir(repo_root)

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

    # 2) 在数据仓库里定位交易日目录（自动/指定）
    trade_date, trade_dir = _resolve_trade_date_dir(settings)
    logger.info(f"定位到交易日: {trade_date}，目录: {trade_dir}")

    # 3) 读取涨停池并打分
    df_pool = load_limitup_pool(trade_dir, settings.snapshot_candidates)
    logger.info(f"涨停池载入成功: rows={len(df_pool)}, cols={len(df_pool.columns)}")

    df_scored = score_stocks(df_pool, settings.weights)
    df_prob = to_probability(df_scored, settings.prob_map)

    # 4) 输出 TopN
    top_df = (
        df_prob.sort_values(["score"], ascending=False)
        .head(settings.top_n)
        .reset_index(drop=True)
    )

    predict_json, predict_md = write_outputs(trade_date, top_df, settings)
    logger.info(f"输出完成: {predict_json}")
    logger.info(f"输出完成: {predict_md}")

    # 5) 追加历史
    append_history(trade_date, top_df, settings)
    logger.info(f"历史沉淀完成: {settings.output.history_append_jsonl}")

    logger.info("运行成功 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
