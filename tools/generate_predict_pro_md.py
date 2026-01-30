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
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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
        row = {}
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


def build_pro_md(
    date_str: str,
    source_md_path: Path,
    source_title: str,
    table_html: str,
    rows: List[Dict[str, str]],
    repo_hint: str = "a-share-top10-engine",
) -> str:
    """
    生成专业版 MD 内容：包含头部元信息 + 原表格（HTML）+ Markdown 表格 + 扩展栏目占位
    """
    # 尽量从 rows 推测来源文件
    source_files = sorted({r.get("source_file", "").strip() for r in rows if r.get("source_file")})
    source_files_str = ", ".join(source_files) if source_files else "N/A"

    # 生成时间（北京时间）
    gen_time_bj = now_bj_str()

    # 推荐 Top3（按 prob 降序 / 若缺失则按 score）
    def _to_float(x: str) -> float:
        try:
            return float(x)
        except Exception:
            return float("-inf")

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
    md.append("## ⑤ 解释与标签（预留，后续接入更丰富因子）")
    md.append("")
    md.append("- 入选原因摘要：_待接入（如：板块热度/情绪过滤/连板结构/龙虎榜/炸板等）_")
    md.append("- 板块/概念归因：_待接入_")
    md.append("- 风险提示：_待接入（如：停牌/一字板/新股/异常波动）_")
    md.append("")
    md.append("## ⑥ 回测与命中统计（预留）")
    md.append("")
    md.append("- 过去5日 Top10 命中：_待接入_")
    md.append("- 过去20日 Top10 命中：_待接入_")
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
    ap.add_argument("--repo-hint", default="a-share-top10-engine", help="仓库/引擎提示信息")
    args = ap.parse_args()

    repo_root = Path(".").resolve()

    predict_dir = repo_root / args.predict_dir
    out_dir = repo_root / args.out_dir
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
