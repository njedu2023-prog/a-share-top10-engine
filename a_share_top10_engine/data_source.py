# a_share_top10_engine/data_source.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd


DEFAULT_TABLES = [
    "limit_list_d",
    "limit_break_d",
    "daily",
    "stk_limit",
    "daily_basic",
    "stock_basic",
    "namechange",
    "top_list",
    "moneyflow_hsgt",
    "hot_boards",
    "limit_up_tags",
]


@dataclass
class DataSource:
    """
    统一数据入口：
    - local_root: 指向 a-share-top3-data 仓库根目录（本地模式）
    - github_raw_base: 线上模式的 raw base（例如: https://raw.githubusercontent.com/<user>/<repo>/<branch>）
    二选一即可。
    """
    local_root: Optional[Path] = None
    github_raw_base: Optional[str] = None  # 不要以 / 结尾
    branch: str = "main"

    def _local_path(self, table: str, trade_date: Optional[str]) -> Path:
        assert self.local_root is not None
        if trade_date:
            year = trade_date[:4]
            return self.local_root / "data" / "raw" / year / trade_date / f"{table}.csv"
        return self.local_root / "data" / "latest" / f"{table}.csv"

    def _remote_url(self, table: str, trade_date: Optional[str]) -> str:
        assert self.github_raw_base is not None
        if trade_date:
            year = trade_date[:4]
            return f"{self.github_raw_base}/data/raw/{year}/{trade_date}/{table}.csv"
        return f"{self.github_raw_base}/data/latest/{table}.csv"

    def load_table(self, table: str, trade_date: Optional[str] = None) -> pd.DataFrame:
        """
        读取指定表；读不到/空文件 -> 返回空 DataFrame（保证引擎不崩）
        """
        try:
            if self.local_root is not None:
                p = self._local_path(table, trade_date)
                if not p.exists() or p.stat().st_size == 0:
                    return pd.DataFrame()
                return pd.read_csv(p, dtype=str, encoding="utf-8-sig")

            if self.github_raw_base is not None:
                url = self._remote_url(table, trade_date)
                return pd.read_csv(url, dtype=str)

            raise RuntimeError("DataSource requires either local_root or github_raw_base")
        except Exception:
            return pd.DataFrame()
