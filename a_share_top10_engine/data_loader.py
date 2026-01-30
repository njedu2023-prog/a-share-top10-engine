# a_share_top10_engine/data_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .data_source import DataSource


# ============================
# Local Snapshot Location (Optional)
# ============================

@dataclass
class SnapshotPaths:
    """
    定义：本地数据仓库快照的标准路径结构
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
        return Path(self.warehouse_root) / self.repo_name / self.raw_dir / year / trade_date


# ============================
# Core Data Loader
# ============================

class DataLoader:
    """
    数据读取层（全系统唯一入口）
    不做业务逻辑，只负责：
    - 定位快照（本地/线上）
    - 读取 CSV（空文件/缺失 -> 空 DataFrame）
    - 返回 DataFrame
    """

    def __init__(
        self,
        trade_date: str,
        *,
        # 本地模式：传入 paths 或 local_root（二选一都行）
        paths: SnapshotPaths | None = None,
        local_root: Path | str | None = None,
        # 线上模式：传入 github_raw_base
        github_raw_base: str | None = None,
    ):
        self.trade_date = trade_date

        # 1) 显式 local_root 优先
        if local_root is not None:
            local_root = Path(local_root)
            self.ds = DataSource(local_root=local_root)

        # 2) 否则如果指定了 github_raw_base，则走线上
        elif github_raw_base is not None:
            self.ds = DataSource(github_raw_base=github_raw_base)

        # 3) 否则默认走你原来的 _warehouse 快照结构
        else:
            p = (paths or SnapshotPaths()).snapshot_dir(trade_date)
            if not p.exists():
                raise FileNotFoundError(
                    f"[DataLoader] 快照目录不存在: {p}\n"
                    f"请确认仓库 checkout 是否成功，或改用 github_raw_base / local_root。"
                )
            # 注意：DataSource 的 local_root 是仓库根目录，所以这里要回退到 repo 根
            # p = _warehouse/a-share-top3-data/data/raw/YYYY/YYYYMMDD
            # repo_root = _warehouse/a-share-top3-data
            repo_root = p.parents[3]
            self.ds = DataSource(local_root=repo_root)

    # -------------------------
    # Internal Reader
    # -------------------------
    def _read_table(self, table: str, trade_date: Optional[str] = None) -> pd.DataFrame:
        """
        统一读取入口：
        - 读不到/空文件/异常 -> 空 DataFrame
        """
        return self.ds.load_table(table=table, trade_date=trade_date or self.trade_date)

    # -------------------------
    # Public Load Functions
    # -------------------------

    def load_limit_list(self) -> pd.DataFrame:
        """
        涨停池（首板候选入口）
        文件：limit_list_d.csv
        """
        return self._read_table("limit_list_d")

    def load_limit_break(self) -> pd.DataFrame:
        """
        炸板池（炸板次数/炸板率）
        文件：limit_break_d.csv
        """
        return self._read_table("limit_break_d")

    def load_daily(self) -> pd.DataFrame:
        """
        日线行情（成交额/放量）
        文件：daily.csv
        """
        return self._read_table("daily")

    def load_limit_up_tags(self) -> pd.DataFrame:
        """
        涨停题材标签（是否核心题材）
        文件：limit_up_tags.csv
        """
        return self._read_table("limit_up_tags")

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
    # 本地默认：_warehouse/a-share-top3-data/...
    loader = DataLoader(trade_date="20260129")
    dfs = loader.load_all_core()

    for name, df in dfs.items():
        print("\n======================")
        print("Loaded:", name)
        print("Shape:", df.shape)
        print(df.head())

    # 线上 raw 示例（需要你换成真实地址）
    # github_raw_base = "https://raw.githubusercontent.com/<user>/a-share-top3-data/main"
    # loader2 = DataLoader(trade_date="20260129", github_raw_base=github_raw_base)
    # print(loader2.load_limit_list().head())
