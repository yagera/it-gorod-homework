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


HEADER_MARKER = "Текст документа"
HEADER_EXTRA = 1000  # запас символов головы на шапку при `strip_header` (шапка ≈ 150–400 символов)
TAIL_SEP = " "


def compose_text(
    head_str: str, tail_str: str, total_len: int, head: int, tail: int, strip: bool
) -> str:
    """Собирает «голова + хвост» (и опционально без шапки) из прочитанных кусков документа.

    `head_str` — первые `head` (+`HEADER_EXTRA` при `strip`) символов, `tail_str` — последние
    `tail`, `total_len` — длина полного текста. Короткий документ (≤ head + tail после снятия
    шапки) берётся целиком, без дублей на стыке. Маркер шапки ищется только в `head_str`.
    """
    cut = 0
    if strip:
        i = head_str.find(HEADER_MARKER)
        if i >= 0:
            cut = i + len(HEADER_MARKER)
            cut += len(head_str[cut:]) - len(head_str[cut:].lstrip())
    rest = total_len - len(head_str)  # символов за пределами головы
    if rest <= 0:
        full = head_str
    elif tail and rest <= tail:
        full = head_str + tail_str[len(tail_str) - rest :]
    else:
        full = None  # между головой и хвостом есть пропуск
    if full is not None:
        t = full[cut:]
        if tail == 0:
            return t[:head]
        return t if len(t) <= head + tail else t[:head] + TAIL_SEP + t[-tail:]
    head_part = head_str[cut:][:head]
    return head_part + TAIL_SEP + tail_str if tail else head_part


def iter_t2_batches(
    path: str | Path,
    columns: list[str],
    batch_size: int,
    max_chars: int,
    limit: int | None = None,
    tail_chars: int = 0,
    strip_header: bool = False,
) -> Iterator[pd.DataFrame]:
    """Батчи parquet T2; `text` усекается до `max_chars` (голова документа, там шапка)
    ещё на стороне Arrow, до перехода в pandas. Чтение останавливается после `limit` строк.
    `tail_chars` > 0 добавляет последние символы документа (срез Arrow с отрицательным
    началом), `strip_header` убирает шапку до «Текст документа»: в pandas попадают только
    куски длиной ~ `max_chars` + запас и `tail_chars`."""
    remaining = limit
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        if remaining is not None:
            if remaining <= 0:
                return
            batch = batch.slice(0, remaining)
            remaining -= batch.num_rows
        if "text" in columns:
            idx = batch.schema.get_field_index("text")
            col = batch.column(idx)
            if not tail_chars and not strip_header:
                text = pc.utf8_slice_codeunits(col, 0, max_chars)
            else:
                head_n = max_chars + (HEADER_EXTRA if strip_header else 0)
                heads = pc.utf8_slice_codeunits(col, 0, head_n).to_pylist()
                totals = pc.utf8_length(col).to_pylist()
                tails = (
                    pc.utf8_slice_codeunits(col, -tail_chars).to_pylist()
                    if tail_chars
                    else [""] * len(heads)
                )
                text = pa.array(
                    [
                        compose_text(h, t, n, max_chars, tail_chars, strip_header)
                        for h, t, n in zip(heads, tails, totals, strict=True)
                    ],
                    type=pa.string(),
                )
            batch = batch.set_column(idx, "text", text)
        yield pa.Table.from_batches([batch]).to_pandas()


def cache_path(cfg: T2Config, split: Literal["train", "test"], smoke: bool) -> Path:
    """Имя кэша включает `max_chars` и (для smoke-train) число строк — старый кэш не подхватится."""
    suffix = f"_smoke{cfg.smoke_n_train_rows}" if smoke and split == "train" else ""
    mode = (f"_t{cfg.tail_chars}" if cfg.tail_chars else "") + (
        "_nohdr" if cfg.strip_header else ""
    )
    return cfg.paths.processed_dir / f"t2_{split}_c{cfg.max_chars}{mode}{suffix}.parquet"


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
    frames = iter_t2_batches(
        source,
        columns,
        cfg.batch_size,
        cfg.max_chars,
        limit,
        cfg.tail_chars,
        cfg.strip_header,
    )
    df = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_t2_codes(path: str | Path) -> set[str]:
    """Допустимые коды — все уникальные `code` train (читается только эта колонка)."""
    return set(pq.read_table(path, columns=["code"]).column("code").to_pylist())


def read_t1_raw(path: str | Path) -> pd.DataFrame:
    """CSV T1 как есть (с пустыми строками и float-ключами) — только для EDA-проверок качества."""
    return pd.read_csv(path)
