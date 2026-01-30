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


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_actual_limitup_codes(warehouse_dir: Path, trade_date: str) -> Set[str]:
    """
    从数据仓库快照里取“当日真实涨停名单”。
    兼容两种常见文件：
      - stk_limit.csv（推荐）
      - limit_list_d.csv
    仅依赖最基础字段：trade_date + ts_code(或 code)
    """
    # 你仓库具体文件名可能在不同子目录里；这里做最稳健的“多路径候选”。
    candidates = [
        warehouse_dir / trade_date / "stk_limit.csv",
        warehouse_dir / trade_date / "limit_list_d.csv",
        warehouse_dir / trade_date / "limit_list.csv",
    ]

    import pandas as pd  # 你项目已依赖 pandas

    for fp in candidates:
        if not fp.exists():
            continue
        df = pd.read_csv(fp)

        # 兼容字段
        cols = set(df.columns)
        # 常见：trade_date/ts_code
        if "trade_date" in cols:
            df = df[df["trade_date"].astype(str) == str(trade_date)]

        code_col = None
        for c in ["ts_code", "code", "股票代码", "symbol"]:
            if c in cols:
                code_col = c
                break
        if not code_col:
            continue

        codes = set(df[code_col].astype(str).tolist())
        # 清洗一下：去空
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
    # 1) predict_latest.json
    p_latest = outputs_dir / "predict_latest.json"
    obj = _safe_read_json(p_latest)
    if obj and str(obj.get("target_trade_date", "")) == str(verify_date):
        return obj

    # 2) 搜索 predict/*.json
    pred_dir = outputs_dir / "predict"
    if not pred_dir.exists():
        return None

    matches: List[Tuple[str, Path]] = []
    for fp in sorted(pred_dir.glob("predict_*.json")):
        o = _safe_read_json(fp)
        if not o:
            continue
        if str(o.get("target_trade_date", "")) == str(verify_date):
            # 用文件名做粗略排序键
            matches.append((fp.name, fp))

    if not matches:
        return None

    # 取最后一条（最新）
    return _safe_read_json(matches[-1][1])


def run_backtest_loop(
    *,
    outputs_dir: Path,
    warehouse_dir: Path,
    verify_date: str,
    history_jsonl: Path,
    top_n_default: int = 10,
) -> Optional[BacktestResult]:
    """
    用 verify_date(今天) 的真实涨停结果，回测一条“目标日=今天”的预测记录。
    """
    pred = _load_pending_prediction(outputs_dir, verify_date)
    if not pred:
        return None

    predict_date = str(pred.get("predict_date", ""))
    target_date = str(pred.get("target_trade_date", ""))
    top_list = pred.get("top_list") or pred.get("top_n") or []
    if not isinstance(top_list, list) or not top_list:
        return None

    # 取代码字段
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

    # 真实涨停
    actual_codes = _load_actual_limitup_codes(warehouse_dir, verify_date)

    hit = sorted(list(set(predicted_codes) & set(actual_codes)))
    top_n = int(pred.get("top_n", top_n_default))
    hit_count = len(hit)
    hit_rate = round(hit_count / max(1, min(top_n, len(predicted_codes))), 4)

    result = BacktestResult(
        predict_date=predict_date,
        target_date=target_date,
        verify_date=str(verify_date),
        top_n=min(top_n, len(predicted_codes)),
        hit_count=hit_count,
        hit_rate=hit_rate,
        hit_codes=hit,
        predicted_codes=predicted_codes[: min(top_n, len(predicted_codes))],
    )

    # 写入 history.jsonl（核心闭环）
    _append_jsonl(
        history_jsonl,
        {
            "predict_date": result.predict_date,
            "target_date": result.target_date,
            "verify_date": result.verify_date,
            "top_n": result.top_n,
            "hit_count": result.hit_count,
            "hit_rate": result.hit_rate,
            "hit_codes": result.hit_codes,
            "predicted_codes": result.predicted_codes,
        },
    )

    return result
