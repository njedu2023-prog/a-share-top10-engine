# a_share_top10_engine/data_loader.py
from __future__ import annotations

import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict


# ============================
# Data Snapshot Location
# ============================

@dataclass
class SnapshotPaths:
    """
    定义：数据仓库快照的标准路径结构
    默认约定：
    _warehouse/a-share-top3-data/data/raw/YYYY/YYYYMMDD/
    """

    warehouse_root: str = "_warehouse"
    repo_name: str = "a-share-top3-data"
    raw_dir: str = "data/raw"

    def snapshot_dir(self, trade_date: str) -> Path:
        """
        输入: trade_date = '20260129'
        输出: Path('_warehouse/a-share-top3-data/data/raw/2026/20260129/')
        """
        year = trade_date[:4]
        return (
            Path(self.warehouse_root)
            / self.repo_name
            / self.raw_dir
            / year
            / trade_date
        )


# ============================
# Core Data Loader
# ============================

class DataLoader:
    """
    数据读取层（全系统唯一入口）
    不做业务逻辑，只负责：
    - 找到快照目录
    - 读取 CSV
    - 返回 DataFrame
    """

    def __init__(self, trade_date: str, paths: SnapshotPaths | None = None):
        self.trade_date = trade_date
        self.paths = paths or SnapshotPaths()

        self.base_dir = self.paths.snapshot_dir(trade_date)

        if not self.base_dir.exists():
            raise FileNotFoundError(
                f"[DataLoader] 快照目录不存在: {self.base_dir}\n"
                f"请确认仓库 checkout 是否成功。"
            )

    # -------------------------
    # Internal CSV Reader
    # -------------------------
    def _read_csv(self, filename: str) -> pd.DataFrame:
        """
        兼容“文件存在但为空”的情况：
        - 0字节空文件：直接返回空 DataFrame（避免 pandas.errors.EmptyDataError）
        - 非 0 字节但内容不含列：捕获 EmptyDataError，返回空 DataFrame
        """
        path = self.base_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"[DataLoader] 文件不存在: {path}\n"
                f"请检查数据仓库是否包含该文件。"
            )

        # 关键修复点：空文件直接返回空 df，不让系统崩掉
        try:
            if path.stat().st_size == 0:
                return pd.DataFrame()
        except OSError:
            # 极端情况下 stat 失败，继续走 read_csv，并在下面兜底
            pass

        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    # -------------------------
    # Public Load Functions
    # -------------------------

    def load_limit_list(self) -> pd.DataFrame:
        """
        涨停池（首板候选入口）
        文件：limit_list_d.csv
        """
        return self._read_csv("limit_list_d.csv")

    def load_limit_break(self) -> pd.DataFrame:
        """
        炸板池（炸板次数/炸板率）
        文件：limit_break_d.csv
        """
        return self._read_csv("limit_break_d.csv")

    def load_daily(self) -> pd.DataFrame:
        """
        日线行情（成交额/放量）
        文件：daily.csv
        """
        return self._read_csv("daily.csv")

    def load_limit_up_tags(self) -> pd.DataFrame:
        """
        涨停题材标签（是否核心题材）
        文件：limit_up_tags.csv
        """
        return self._read_csv("limit_up_tags.csv")

    # -------------------------
    # One-shot Loading (Optional)
    # -------------------------
    def load_all_core(self) -> Dict[str, pd.DataFrame]:
        """
        一次性加载核心四表（模块②③④必备）
        """
        return {
            "limit_list": self.load_limit_list(),
            "limit_break": self.load_limit_break(),
            "daily": self.load_daily(),
            "tags": self.load_limit_up_tags(),
        }


# ============================
# Quick Test入口（本地调试用）
# ============================

if __name__ == "__main__":
    loader = DataLoader(trade_date="20260129")

    dfs = loader.load_all_core()

    for name, df in dfs.items():
        print("\n======================")
        print("Loaded:", name)
        print("Shape:", df.shape)
        print(df.head())
