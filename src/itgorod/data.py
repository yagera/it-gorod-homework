"""Чтение данных: T1 (CSV) и T2 (parquet батчами с усечением и кэшем)."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from itgorod.config import T2Config


def read_t1(path: str | Path) -> pd.DataFrame:
    """CSV T1; полностью пустые строки выбрасываются, department_id и year → int."""
    df = pd.read_csv(path).dropna(how="all")
    return df.astype({"department_id": "int64", "year": "int64"})


def iter_t2_batches(
    path: str | Path,
    columns: list[str],
    batch_size: int,
    max_chars: int,
    limit: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Батчи parquet T2; `text` усекается до `max_chars` (голова документа, там шапка)
    ещё на стороне Arrow, до перехода в pandas. Чтение останавливается после `limit` строк."""
    remaining = limit
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        if remaining is not None:
            if remaining <= 0:
                return
            batch = batch.slice(0, remaining)
            remaining -= batch.num_rows
        if "text" in columns:
            idx = batch.schema.get_field_index("text")
            truncated = pc.utf8_slice_codeunits(batch.column(idx), 0, max_chars)
            batch = batch.set_column(idx, "text", truncated)
        yield pa.Table.from_batches([batch]).to_pandas()


def cache_path(cfg: T2Config, split: Literal["train", "test"], smoke: bool) -> Path:
    """Имя кэша включает `max_chars` и (для smoke-train) число строк — старый кэш не подхватится."""
    suffix = f"_smoke{cfg.smoke_n_train_rows}" if smoke and split == "train" else ""
    return cfg.paths.processed_dir / f"t2_{split}_c{cfg.max_chars}{suffix}.parquet"


def build_t2_cache(
    cfg: T2Config, split: Literal["train", "test"], smoke: bool, force: bool = False
) -> Path:
    """Усечённые тексты → `data/processed`. Test всегда целиком; smoke ограничивает только train."""
    path = cache_path(cfg, split, smoke)
    if path.exists() and not force:
        return path
    source = cfg.paths.train if split == "train" else cfg.paths.test
    columns = ["text", "code"] if split == "train" else ["row_id", "text"]
    limit = cfg.smoke_n_train_rows if smoke and split == "train" else None
    frames = iter_t2_batches(source, columns, cfg.batch_size, cfg.max_chars, limit)
    df = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_t2_codes(path: str | Path) -> set[str]:
    """Допустимые коды — все уникальные `code` train (читается только эта колонка)."""
    return set(pq.read_table(path, columns=["code"]).column("code").to_pylist())
