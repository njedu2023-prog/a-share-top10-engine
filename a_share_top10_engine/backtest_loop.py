from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class BacktestResult:
    predict_date: str
    target_date: str
    verify_date: str
    top_n: int
    hit_count: int
    hit_rate: float
    hit_codes: List[str]
    predicted_codes: List[str]


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _resolve_trade_dir(warehouse_root: Path, trade_date: str) -> Optional[Path]:
    """
    自动定位数据仓库中某个交易日目录，兼容多种常见结构：
      - _warehouse/YYYYMMDD
      - _warehouse/data/YYYYMMDD
      - _warehouse/data/raw/YYYYMMDD
      - _warehouse/data/raw/YYYY/YYYYMMDD   （你 workflow 当前优先匹配的就是这个）
      - _warehouse/YYYY/YYYYMMDD
    """
    y = str(trade_date)[:4]
    candidates = [
        warehouse_root / str(trade_date),
        warehouse_root / "data" / str(trade_date),
        warehouse_root / "data" / "raw" / str(trade_date),
        warehouse_root / "data" / "raw" / y / str(trade_date),
        warehouse_root / y / str(trade_date),
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


def _load_actual_limitup_codes(trade_dir: Path, trade_date: str) -> Set[str]:
    """
    从 trade_dir 快照里取“当日真实涨停名单”。
    兼容两种常见文件：
      - stk_limit.csv（推荐）
      - limit_list_d.csv / limit_list.csv
    仅依赖最基础字段：trade_date + ts_code(或 code)
    """
    candidates = [
        trade_dir / "stk_limit.csv",
        trade_dir / "limit_list_d.csv",
        trade_dir / "limit_list.csv",
    ]

    import pandas as pd  # 项目已依赖 pandas

    for fp in candidates:
        if not fp.exists():
            continue

        df = pd.read_csv(fp)
        cols = set(df.columns)

        # 过滤 trade_date（若存在）
        if "trade_date" in cols:
            df = df[df["trade_date"].astype(str) == str(trade_date)]

        # 识别 code 列
        code_col = None
        for c in ["ts_code", "code", "股票代码", "symbol"]:
            if c in cols:
                code_col = c
                break
        if not code_col:
            continue

        codes = set(df[code_col].astype(str).tolist())
        codes = {c.strip() for c in codes if isinstance(c, str) and c.strip()}
        if codes:
            return codes

    return set()


def _load_pending_prediction(outputs_dir: Path, verify_date: str) -> Optional[Dict[str, Any]]:
    """
    找到“目标日 = verify_date”的那条预测记录（待回测）。
    规则：
      1) 优先：outputs/predict_latest.json
      2) 其次：outputs/predict/predict_*.json 中 target_trade_date == verify_date 的最新一条
    """
    p_latest = outputs_dir / "predict_latest.json"
    obj = _safe_read_json(p_latest)
    if obj and str(obj.get("target_trade_date", "")) == str(verify_date):
        return obj

    pred_dir = outputs_dir / "predict"
    if not pred_dir.exists():
        return None

    matches: List[Tuple[str, Path]] = []
    for fp in sorted(pred_dir.glob("predict_*.json")):
        o = _safe_read_json(fp)
        if not o:
            continue
        if str(o.get("target_trade_date", "")) == str(verify_date):
            matches.append((fp.name, fp))

    if not matches:
        return None

    return _safe_read_json(matches[-1][1])


def run_backtest_loop(
    *,
    outputs_dir: Path,
    warehouse_root: Path,
    verify_date: str,
    history_jsonl: Path,
    top_n_default: int = 10,
    emit_files: bool = True,
) -> Optional[BacktestResult]:
    """
    用 verify_date(今天) 的真实涨停结果，回测一条“目标日=今天”的预测记录。
    成功时：
      - 追加 outputs/history/backtest_history.jsonl
      - 写 outputs/history/backtest_{verify_date}.json / backtest_latest.json
      - 写 outputs/history/backtest_{verify_date}.md（方便肉眼查看）
    """
    pred = _load_pending_prediction(outputs_dir, verify_date)
    if not pred:
        return None

    trade_dir = _resolve_trade_dir(warehouse_root, verify_date)
    if not trade_dir:
        return None

    predict_date = str(pred.get("predict_date", ""))
    target_date = str(pred.get("target_trade_date", ""))
    top_list = pred.get("top_list") or pred.get("top_n") or []
    if not isinstance(top_list, list) or not top_list:
        return None

    predicted_codes: List[str] = []
    for item in top_list:
        if isinstance(item, dict):
            code = item.get("ts_code") or item.get("code") or item.get("symbol")
            if code:
                predicted_codes.append(str(code))
        elif isinstance(item, str):
            predicted_codes.append(item)

    predicted_codes = [c.strip() for c in predicted_codes if c and isinstance(c, str)]
    if not predicted_codes:
        return None

    actual_codes = _load_actual_limitup_codes(trade_dir, verify_date)

    hit = sorted(list(set(predicted_codes) & set(actual_codes)))
    top_n = int(pred.get("top_n", top_n_default))
    top_n_effective = min(top_n, len(predicted_codes))
    hit_count = len(hit)
    hit_rate = round(hit_count / max(1, top_n_effective), 4)

    result = BacktestResult(
        predict_date=predict_date,
        target_date=target_date,
        verify_date=str(verify_date),
        top_n=top_n_effective,
        hit_count=hit_count,
        hit_rate=hit_rate,
        hit_codes=hit,
        predicted_codes=predicted_codes[:top_n_effective],
    )

    payload = {
        "predict_date": result.predict_date,
        "target_date": result.target_date,
        "verify_date": result.verify_date,
        "top_n": result.top_n,
        "hit_count": result.hit_count,
        "hit_rate": result.hit_rate,
        "hit_codes": result.hit_codes,
        "predicted_codes": result.predicted_codes,
        "trade_dir": str(trade_dir),
    }

    _append_jsonl(history_jsonl, payload)

    if emit_files:
        out_dir = history_jsonl.parent
        _write_json(out_dir / f"backtest_{verify_date}.json", payload)
        _write_json(out_dir / "backtest_latest.json", payload)

        md = []
        md.append(f"# Backtest ({verify_date})\n")
        md.append(f"- predict_date: `{result.predict_date}`")
        md.append(f"- target_date: `{result.target_date}`")
        md.append(f"- verify_date: `{result.verify_date}`")
        md.append(f"- top_n: `{result.top_n}`")
        md.append(f"- hit_count: `{result.hit_count}`")
        md.append(f"- hit_rate: `{result.hit_rate}`")
        md.append("")
        md.append("## Hit codes")
        md.append(", ".join(result.hit_codes) if result.hit_codes else "(none)")
        md.append("")
        md.append("## Predicted codes")
        md.append(", ".join(result.predicted_codes) if result.predicted_codes else "(none)")
        md.append("")
        (out_dir / f"backtest_{verify_date}.md").write_text("\n".join(md), encoding="utf-8")

    return result
