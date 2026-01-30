#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_predict_pro_md.py

旁路增强版报告生成器（不修改原有 predict_YYYYMMDD.md 的生成链路）
输入：outputs/predict/predict_YYYYMMDD.md（旧版日报）
输出：
  - outputs/predict_pro/predict_pro_YYYYMMDD.md（专业版日报）
  - outputs/predict_pro/latest.md（固定入口，指向最近一次专业版内容）

用法：
  1) 自动识别最新日期：
     python tools/generate_predict_pro_md.py

  2) 指定日期：
     python tools/generate_predict_pro_md.py --date 20260129

  3) 指定输入文件：
     python tools/generate_predict_pro_md.py --in outputs/predict/predict_20260129.md
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# 北京时间时区（UTC+8）
TZ_BJ = timezone(timedelta(hours=8))


def now_bj_str() -> str:
    return datetime.now(TZ_BJ).strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_latest_predict_md(predict_dir: Path) -> Optional[Path]:
    """
    在 outputs/predict/ 里找最新的 predict_YYYYMMDD.md
    """
    if not predict_dir.exists():
        return None

    candidates = []
    for fp in predict_dir.glob("predict_*.md"):
        m = re.search(r"predict_(\d{8})\.md$", fp.name)
        if m:
            candidates.append((m.group(1), fp))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def extract_date_from_filename(fp: Path) -> Optional[str]:
    m = re.search(r"predict_(\d{8})\.md$", fp.name)
    return m.group(1) if m else None


def read_text(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="replace")


def write_text(fp: Path, text: str) -> None:
    fp.write_text(text, encoding="utf-8")


def read_json(fp: Path) -> Any:
    return json.loads(fp.read_text(encoding="utf-8", errors="replace"))


def parse_title_and_table(md: str) -> Tuple[str, str]:
    """
    从旧 md 中提取标题行（如果有）与 <table>...</table> 块。
    旧文件现在是：
      Top10 Prediction (YYYYMMDD)

      <table>...</table>

    我们会尽量稳健地找 table。
    """
    lines = [ln.rstrip("\n") for ln in md.splitlines()]
    title = ""
    for ln in lines:
        if ln.strip():
            title = ln.strip()
            break

    # 找 table 块
    m = re.search(r"(<table>.*?</table>)", md, flags=re.S)
    if not m:
        # 兼容 markdown 表格（未来如果改了）
        # 简单兜底：返回原始内容
        return title or "Top10 Prediction", md.strip()

    table_html = m.group(1).strip()
    return title or "Top10 Prediction", table_html


def parse_rows_from_table_html(table_html: str) -> List[Dict[str, str]]:
    """
    从 HTML table 里解析出行数据（不引入第三方库，使用正则简解析）
    输出每行 dict：{ts_code,name,score,prob,source_file}
    """
    # 提取所有 <tr>...</tr>
    trs = re.findall(r"<tr>(.*?)</tr>", table_html, flags=re.S)
    if not trs:
        return []

    # 第一行是表头
    ths = re.findall(r"<th>(.*?)</th>", trs[0], flags=re.S)
    headers = [strip_html(x) for x in ths] if ths else []

    rows: List[Dict[str, str]] = []
    for tr in trs[1:]:
        tds = re.findall(r"<td>(.*?)</td>", tr, flags=re.S)
        if not tds:
            continue
        values = [strip_html(x) for x in tds]
        row: Dict[str, str] = {}
        if headers and len(headers) == len(values):
            for h, v in zip(headers, values):
                row[h] = v
        else:
            # 兜底按固定顺序
            keys = ["ts_code", "name", "score", "prob", "source_file"]
            for i, k in enumerate(keys):
                if i < len(values):
                    row[k] = values[i]
        rows.append(row)
    return rows


def strip_html(s: str) -> str:
    s = re.sub(r"<.*?>", "", s, flags=re.S)  # 去标签
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    return s.strip()


def to_markdown_table(rows: List[Dict[str, str]]) -> str:
    """
    生成 Markdown 表格（更利于阅读/复制）
    """
    if not rows:
        return "_（未解析到表格内容）_"

    headers = ["ts_code", "name", "score", "prob", "source_file"]
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r.get(h, "") for h in headers) + " |")
    return "\n".join(out)


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("-inf")


# -----------------------------
# ⑤ 解释与标签：启发式归因
# -----------------------------
_KEYWORD_SECTORS: List[Tuple[str, str]] = [
    # 贵金属/有色
    ("黄金", "贵金属/黄金"),
    ("铝", "有色金属/铝"),
    ("锌", "有色金属/锌"),
    ("铜", "有色金属/铜"),
    ("矿", "资源/采矿"),
    ("锂", "新能源/锂电上游"),
    ("稀土", "有色金属/稀土"),
    # 能源/化工
    ("能源", "能源/油气"),
    ("石油", "能源/油气"),
    ("煤", "能源/煤炭"),
    ("化工", "化工"),
    ("化学", "化工"),
    # 农业/食品
    ("种业", "农业/种业"),
    ("农", "农业"),
    ("粮", "农业/粮食"),
    # 电力/光伏
    ("光伏", "新能源/光伏"),
    ("新能", "新能源"),
    ("电", "电力/设备"),
    # 基建/工程
    ("工程", "基建/工程"),
    ("建设", "基建/工程"),
    ("诚", "（可能）基建/工程"),
]

# source_file -> 事件/信号
_SOURCE_REASON: List[Tuple[str, str]] = [
    ("limit_list", "来自涨停/强势池（短线情绪强，资金关注度高）"),
    ("dragon", "来自龙虎榜/大单异动（资金博弈信号）"),
    ("hot", "来自热度/情绪筛选（短线关注度）"),
    ("north", "来自北向/外资偏好（资金流向信号）"),
]


def infer_sectors(rows: List[Dict[str, str]], topk: int = 5) -> List[str]:
    """
    用股票名称关键词做“热门板块/概念”启发式归因（无外部数据可直接跑）。
    返回：按出现次数排序的 sector 列表。
    """
    if not rows:
        return []

    counts: Dict[str, int] = {}
    for r in rows:
        name = (r.get("name") or "").strip()
        hit_any = False
        for kw, sector in _KEYWORD_SECTORS:
            if kw and kw in name:
                counts[sector] = counts.get(sector, 0) + 1
                hit_any = True
        if not hit_any and name:
            counts["其他/未识别"] = counts.get("其他/未识别", 0) + 1

    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [f"{k}（{v}）" for k, v in ranked[:topk]]


def infer_selection_reasons(rows: List[Dict[str, str]]) -> List[str]:
    """
    生成“入选原因”要点（启发式）。
    """
    if not rows:
        return ["未解析到 Top10 数据，无法生成入选原因。"]

    # 1) 信号来源（source_file）
    source_files = sorted({(r.get("source_file") or "").strip() for r in rows if (r.get("source_file") or "").strip()})
    reason_lines: List[str] = []

    if source_files:
        matched = []
        for sf in source_files:
            low = sf.lower()
            for key, desc in _SOURCE_REASON:
                if key in low:
                    matched.append(desc)
        if matched:
            # 去重保持顺序
            seen = set()
            uniq = []
            for x in matched:
                if x not in seen:
                    uniq.append(x)
                    seen.add(x)
            reason_lines.append("信号来源： " + "；".join(uniq))
        else:
            reason_lines.append("信号来源：来自策略筛选结果（source_file 已记录，但未匹配到预置解释）。")
    else:
        reason_lines.append("信号来源：未提供 source_file 字段，暂按综合打分结果展示。")

    # 2) 强信号：Top3 的 prob/score
    ranked = rows[:]
    if any("prob" in r for r in ranked):
        ranked.sort(key=lambda r: _to_float(r.get("prob", "")), reverse=True)
    else:
        ranked.sort(key=lambda r: _to_float(r.get("score", "")), reverse=True)
    top1 = ranked[0] if ranked else {}
    if top1:
        reason_lines.append(
            f"强信号：榜首 `{top1.get('ts_code','')}` {top1.get('name','')} prob={top1.get('prob','')} score={top1.get('score','')}（用于快速聚焦）。"
        )

    # 3) 主题聚合
    sectors = infer_sectors(rows, topk=3)
    if sectors:
        reason_lines.append("主题聚合：Top10 中出现较多的方向为 " + "、".join(sectors) + "。")

    return reason_lines


def infer_risks(rows: List[Dict[str, str]]) -> List[str]:
    """
    生成“风险提示”要点（启发式）。
    """
    if not rows:
        return ["未解析到 Top10 数据，无法生成风险提示。"]

    risks: List[str] = []

    # 涨停/强势池风险
    source_files = " ".join([(r.get("source_file") or "") for r in rows]).lower()
    if "limit_list" in source_files:
        risks.append("短线波动风险：来自涨停/强势池的标的通常波动更大，次日易分化/炸板。")
        risks.append("流动性与滑点：强势股集合竞价/开盘阶段冲击成本可能显著，需控制仓位与下单方式。")

    # 主题拥挤度：prob 高且集中
    probs = [_to_float(r.get("prob", "")) for r in rows if (r.get("prob") or "").strip()]
    probs = [p for p in probs if p > float("-inf")]
    if probs:
        high_cnt = sum(1 for p in probs if p >= 0.75)
        if high_cnt >= 5:
            risks.append("主题拥挤度风险：高概率标的数量较多，可能对应情绪一致预期，需防回撤。")

    # 名称含 ST / 退 / 风险提示
    for r in rows:
        name = (r.get("name") or "").strip()
        if "ST" in name or "退" in name:
            risks.append("个股风险：名单包含 ST/退市相关字样，需核对交易限制与风险公告。")
            break

    # 通用风控
    risks.append("通用提示：本输出为量化筛选结果，不构成投资建议；请结合停复牌/公告/监管风险进行二次过滤。")

    # 去重
    seen = set()
    uniq = []
    for x in risks:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


# -----------------------------
# ⑥ 回测与命中统计：读取 history
# -----------------------------
def _get_record_date(rec: Dict[str, Any]) -> str:
    """
    从回测记录中选择一个“日期字段”，用于排序与窗口截取。
    优先级：verify_date > target_date > target_trade_date > predict_date > date
    """
    for k in ["verify_date", "target_date", "target_trade_date", "predict_date", "date"]:
        v = rec.get(k)
        if isinstance(v, str) and re.fullmatch(r"\d{8}", v.strip()):
            return v.strip()
    return ""


def load_backtest_records(history_dir: Path) -> List[Dict[str, Any]]:
    """
    优先读取 outputs/history/backtest.jsonl（多日累计）。
    兜底读取 backtest_latest.json 或 backtest_*.json（单日或汇总）。
    返回：list[record]
    record 期望含：top_n、hit_count、hit_rate（若无则可由 hit_count/top_n 推导）
    """
    records: List[Dict[str, Any]] = []

    jsonl_fp = history_dir / "backtest.jsonl"
    if jsonl_fp.exists():
        text = jsonl_fp.read_text(encoding="utf-8", errors="replace")
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    records.append(obj)
            except Exception:
                continue

    if records:
        return records

    latest_fp = history_dir / "backtest_latest.json"
    if latest_fp.exists():
        try:
            obj = read_json(latest_fp)
            if isinstance(obj, dict):
                records.append(obj)
        except Exception:
            pass

    # 再兜底：找 backtest_YYYYMMDD.json
    if not records:
        candidates = []
        for fp in history_dir.glob("backtest_*.json"):
            m = re.search(r"backtest_(\d{8})\.json$", fp.name)
            if m:
                candidates.append((m.group(1), fp))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, fp in candidates[:30]:
            try:
                obj = read_json(fp)
                if isinstance(obj, dict):
                    records.append(obj)
            except Exception:
                continue

    return records


def compute_recent_hit_rates(
    history_dir: Path,
    asof_date: str,
    windows: List[int] = [5, 20],
) -> Dict[int, Dict[str, Any]]:
    """
    计算过去 N 日命中率（加权：sum(hit_count)/sum(top_n)）。
    只取 date <= asof_date 的记录，并按日期排序后取最近 N 条。
    返回：{N: {"hit_rate": float, "hit_count": int, "total": int, "days": int, "note": str}}
    """
    out: Dict[int, Dict[str, Any]] = {}
    records = load_backtest_records(history_dir)
    if not records:
        for w in windows:
            out[w] = {"hit_rate": None, "hit_count": 0, "total": 0, "days": 0, "note": "未找到回测历史文件。"}
        return out

    # normalize & filter
    norm: List[Dict[str, Any]] = []
    for rec in records:
        d = _get_record_date(rec)
        if not d:
            continue
        if d <= asof_date:
            norm.append(rec)

    norm.sort(key=lambda r: _get_record_date(r))
    if not norm:
        for w in windows:
            out[w] = {"hit_rate": None, "hit_count": 0, "total": 0, "days": 0, "note": "回测记录日期均晚于报告日期或缺失日期字段。"}
        return out

    for w in windows:
        tail = norm[-w:] if len(norm) >= 1 else []
        if not tail:
            out[w] = {"hit_rate": None, "hit_count": 0, "total": 0, "days": 0, "note": "窗口内无回测记录。"}
            continue

        hit_sum = 0
        total_sum = 0
        used_days = 0

        for rec in tail:
            top_n = rec.get("top_n", None)
            hit_count = rec.get("hit_count", None)

            try:
                top_n_i = int(top_n) if top_n is not None else 0
            except Exception:
                top_n_i = 0
            try:
                hit_i = int(hit_count) if hit_count is not None else 0
            except Exception:
                hit_i = 0

            # 如果没有 top_n/hit_count，尝试用 hit_rate 推导（假设 top_n=10）
            if top_n_i <= 0 and isinstance(rec.get("hit_rate"), (int, float)):
                top_n_i = 10
                hit_i = int(round(float(rec.get("hit_rate")) * top_n_i))

            if top_n_i > 0:
                hit_sum += max(0, hit_i)
                total_sum += top_n_i
                used_days += 1

        if total_sum <= 0:
            out[w] = {"hit_rate": None, "hit_count": 0, "total": 0, "days": used_days, "note": "回测记录缺少 top_n/hit_count。"}
        else:
            out[w] = {
                "hit_rate": hit_sum / total_sum,
                "hit_count": hit_sum,
                "total": total_sum,
                "days": used_days,
                "note": "",
            }

    return out


def build_pro_md(
    date_str: str,
    source_md_path: Path,
    source_title: str,
    table_html: str,
    rows: List[Dict[str, str]],
    repo_hint: str = "a-share-top10-engine",
    history_dir: Optional[Path] = None,
) -> str:
    """
    生成专业版 MD 内容：包含头部元信息 + 原表格（HTML）+ Markdown 表格 + 扩展栏目（⑤⑥自动生成）
    """
    # 尽量从 rows 推测来源文件
    source_files = sorted({r.get("source_file", "").strip() for r in rows if r.get("source_file")})
    source_files_str = ", ".join(source_files) if source_files else "N/A"

    # 生成时间（北京时间）
    gen_time_bj = now_bj_str()

    # 推荐 Top3（按 prob 降序 / 若缺失则按 score）
    ranked = rows[:]
    if ranked and any("prob" in r for r in ranked):
        ranked.sort(key=lambda r: _to_float(r.get("prob", "")), reverse=True)
    else:
        ranked.sort(key=lambda r: _to_float(r.get("score", "")), reverse=True)
    top3 = ranked[:3]

    top3_lines = []
    for i, r in enumerate(top3, start=1):
        top3_lines.append(
            f"{i}. **{r.get('ts_code','')} {r.get('name','')}** ｜prob={r.get('prob','')}｜score={r.get('score','')}"
        )
    top3_block = "\n".join(top3_lines) if top3_lines else "_（无）_"

    # ⑤ 自动生成：入选原因 / 热门板块 / 风险
    reasons = infer_selection_reasons(rows)
    sectors = infer_sectors(rows, topk=5)
    risks = infer_risks(rows)

    # ⑥ 自动生成：过去5/20日命中率
    hit_stats = {}
    if history_dir is not None:
        hit_stats = compute_recent_hit_rates(history_dir=history_dir, asof_date=date_str, windows=[5, 20])

    def _fmt_rate(x: Optional[float]) -> str:
        if x is None:
            return "N/A"
        return f"{x*100:.2f}%"

    # 专业版正文
    md = []
    md.append(f"# Top10 Prediction Pro ({date_str})")
    md.append("")
    md.append("## ① 生成信息（系统元数据）")
    md.append("")
    md.append(f"- 生成时间（北京时间）：**{gen_time_bj}**")
    md.append(f"- 输入文件：`{source_md_path.as_posix()}`")
    md.append(f"- 数据来源文件（来自表格字段）：`{source_files_str}`")
    md.append(f"- 引擎仓库：`{repo_hint}`（旁路增强版，不改原链路）")
    md.append("")
    md.append("## ② Top3 强信号（用于快速决策）")
    md.append("")
    md.append(top3_block)
    md.append("")
    md.append("## ③ Top10 原始表格（保持与旧版一致）")
    md.append("")
    md.append(table_html)
    md.append("")
    md.append("## ④ Top10 Markdown表格（便于复制/二次加工）")
    md.append("")
    md.append(to_markdown_table(rows))
    md.append("")
    md.append("## ⑤ 解释与标签（自动生成）")
    md.append("")
    md.append("### 入选原因")
    md.append("")
    for x in reasons:
        md.append(f"- {x}")
    md.append("")
    md.append("### 热门板块归因（启发式）")
    md.append("")
    if sectors:
        md.append("- " + "、".join(sectors))
    else:
        md.append("- N/A（未识别到板块关键词）")
    md.append("")
    md.append("### 风险提示")
    md.append("")
    for x in risks:
        md.append(f"- {x}")
    md.append("")
    md.append("## ⑥ 回测与命中统计（自动生成）")
    md.append("")
    if not hit_stats:
        md.append("- 过去5日 Top10 命中率：N/A（未提供 history_dir 或未找到回测数据）")
        md.append("- 过去20日 Top10 命中率：N/A（未提供 history_dir 或未找到回测数据）")
    else:
        s5 = hit_stats.get(5, {})
        s20 = hit_stats.get(20, {})

        if s5.get("hit_rate") is None:
            md.append(f"- 过去5日 Top10 命中率：N/A（{s5.get('note','数据不足')}）")
        else:
            md.append(
                f"- 过去5日 Top10 命中率：**{_fmt_rate(s5['hit_rate'])}**（命中 {s5['hit_count']} / 总 {s5['total']}，样本 {s5['days']} 日）"
            )

        if s20.get("hit_rate") is None:
            md.append(f"- 过去20日 Top10 命中率：N/A（{s20.get('note','数据不足')}）")
        else:
            md.append(
                f"- 过去20日 Top10 命中率：**{_fmt_rate(s20['hit_rate'])}**（命中 {s20['hit_count']} / 总 {s20['total']}，样本 {s20['days']} 日）"
            )

    md.append("")
    md.append("---")
    md.append("**说明：**本文件由 `tools/generate_predict_pro_md.py` 生成，属于旁路增强输出，不影响原有 `outputs/predict/` 目录与既有链接。")
    md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="", help="YYYYMMDD，例如 20260129")
    ap.add_argument("--in", dest="in_path", default="", help="指定输入旧版 md 路径")
    ap.add_argument("--predict-dir", default="outputs/predict", help="旧版预测目录（默认 outputs/predict）")
    ap.add_argument("--out-dir", default="outputs/predict_pro", help="专业版输出目录（默认 outputs/predict_pro）")
    ap.add_argument("--history-dir", default="outputs/history", help="回测历史目录（默认 outputs/history）")
    ap.add_argument("--repo-hint", default="a-share-top10-engine", help="仓库/引擎提示信息")
    args = ap.parse_args()

    repo_root = Path(".").resolve()

    predict_dir = repo_root / args.predict_dir
    out_dir = repo_root / args.out_dir
    history_dir = repo_root / args.history_dir
    ensure_dir(out_dir)

    # 1) 确定输入文件
    in_fp: Optional[Path] = None
    date_str = ""

    if args.in_path:
        in_fp = repo_root / args.in_path
        if not in_fp.exists():
            print(f"[ERROR] 输入文件不存在：{in_fp}", file=sys.stderr)
            sys.exit(2)
        date_str = extract_date_from_filename(in_fp) or args.date.strip()
    else:
        if args.date.strip():
            date_str = args.date.strip()
            in_fp = predict_dir / f"predict_{date_str}.md"
            if not in_fp.exists():
                print(f"[ERROR] 未找到指定日期旧版文件：{in_fp}", file=sys.stderr)
                sys.exit(2)
        else:
            in_fp = find_latest_predict_md(predict_dir)
            if not in_fp:
                print(f"[ERROR] 在目录未找到旧版文件：{predict_dir}", file=sys.stderr)
                sys.exit(2)
            date_str = extract_date_from_filename(in_fp) or ""

    if not date_str:
        print("[ERROR] 无法解析日期 YYYYMMDD", file=sys.stderr)
        sys.exit(2)

    # 2) 读取旧 md
    md_old = read_text(in_fp)
    source_title, table_html = parse_title_and_table(md_old)
    rows = parse_rows_from_table_html(table_html)

    # 3) 生成专业版
    pro_md = build_pro_md(
        date_str=date_str,
        source_md_path=in_fp.relative_to(repo_root),
        source_title=source_title,
        table_html=table_html,
        rows=rows,
        repo_hint=args.repo_hint,
        history_dir=history_dir,
    )

    out_fp = out_dir / f"predict_pro_{date_str}.md"
    write_text(out_fp, pro_md)

    # 4) 更新 latest.md（固定入口）
    latest_fp = out_dir / "latest.md"
    write_text(latest_fp, pro_md)

    print(f"[OK] 生成专业版：{out_fp.as_posix()}")
    print(f"[OK] 更新固定入口：{latest_fp.as_posix()}")


if __name__ == "__main__":
    main()
