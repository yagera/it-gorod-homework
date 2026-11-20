"""T2: эмбеддинги документов как опциональный набор признаков (подготовка, без полного кодирования).

Энкодер получает только тексты — меток нет ни в одном аргументе (тест). Эмбеддинги кэшируются
на диск вне git (`data/cache/`), ключ в имени файла: модель, max_tokens, режим шапки, число
документов. Тяжёлые зависимости (torch, transformers) импортируются лениво, группа `dl`.
"""

import dataclasses
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from itgorod.data import HEADER_MARKER

# Кандидаты: пулинг и префикс — по карточкам моделей на Hugging Face.
CANDIDATES: dict[str, dict] = {
    "intfloat/multilingual-e5-base": {"pooling": "mean", "prefix": "passage: ", "params_m": 278},
    "BAAI/bge-m3": {"pooling": "cls", "prefix": "", "params_m": 568},
    "ai-forever/sbert_large_nlu_ru": {"pooling": "mean", "prefix": "", "params_m": 427},
}


@dataclass(frozen=True)
class EmbeddingSpec:
    """Ключ кэша: модель, длина в токенах, режим шапки (`with` / `none`), число документов."""

    model: str
    max_tokens: int
    header: str
    n_docs: int


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


def texts_for_mode(texts: Sequence[str], header: str) -> list[str]:
    """`with` — тексты как есть; `none` — без шапки (после «Текст документа»; без маркера —
    без изменений)."""
    if header == "with":
        return list(texts)
    if header != "none":
        raise ValueError(f"режим шапки: with | none, получено {header!r}")
    out = []
    for t in texts:
        head, sep, body = t.partition(HEADER_MARKER)
        out.append(body.strip() if sep else t)
    return out


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model.replace("/", "--"))


def cache_path(cache_dir: str | Path, spec: EmbeddingSpec) -> Path:
    """`emb_<модель>_t<токены>_h<режим шапки>_n<число документов>.npy` — все части ключа в имени."""
    name = f"emb_{model_slug(spec.model)}_t{spec.max_tokens}_h{spec.header}_n{spec.n_docs}.npy"
    return Path(cache_dir) / name


def encode_with_cache(
    encoder: Encoder,
    texts: Sequence[str],
    spec: EmbeddingSpec,
    cache_dir: str | Path,
    batch_size: int = 32,
) -> np.ndarray:
    """Эмбеддинги текстов (fp16 на диске); при наличии кэша энкодер не вызывается.
    Порядок строк = порядок `texts`; число текстов должно совпадать с `spec.n_docs`."""
    if len(texts) != spec.n_docs:
        raise ValueError(f"число текстов {len(texts)} не равно spec.n_docs {spec.n_docs}")
    path = cache_path(cache_dir, spec)
    if path.exists():
        cached = np.load(path)
        if cached.shape[0] != spec.n_docs:
            raise ValueError(f"кэш {path.name}: {cached.shape[0]} строк, ожидалось {spec.n_docs}")
        return cached
    parts = [
        np.asarray(encoder.encode(list(texts[i : i + batch_size])))
        for i in range(0, len(texts), batch_size)
    ]
    arr = np.concatenate(parts).astype(np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def benchmark_encoder(
    encoder: Encoder,
    texts: Sequence[str],
    batch_size: int = 32,
    clock: Callable[[], float] = time.perf_counter,
    vram_peak_mb: Callable[[], float] = lambda: float("nan"),
) -> dict:
    """Скорость кодирования: документов в секунду, пиковая VRAM, оценка на 50 тыс. документов.
    Кэш не используется (замеряется само кодирование)."""
    t0 = clock()
    for i in range(0, len(texts), batch_size):
        encoder.encode(list(texts[i : i + batch_size]))
    dt = clock() - t0
    dps = len(texts) / dt
    return {
        "n_docs": len(texts),
        "seconds": dt,
        "docs_per_s": dps,
        "peak_vram_mb": vram_peak_mb(),
        "est_50k_min": 50000 / dps / 60,
    }


class HFEncoder:
    """Энкодер на transformers (fp16, GPU). Токен Hugging Face берёт библиотека сама (переменная
    окружения `HF_TOKEN` или файл `hf auth login`); в коде и файлах он не хранится. Меток нет:
    `encode(texts)` — только тексты."""

    def __init__(
        self, model: str, max_tokens: int = 512, device: str = "cuda", batch_size: int = 32
    ):
        import torch  # ленивый импорт: группа dl
        from transformers import AutoModel, AutoTokenizer

        cfg = CANDIDATES[model]
        self._torch = torch
        self.pooling, self.prefix = cfg["pooling"], cfg["prefix"]
        self.max_tokens, self.device = max_tokens, device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, dtype=torch.float16).to(device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        batch = self.tokenizer(
            [self.prefix + t for t in texts],
            padding=True,
            truncation=True,  # голова текста; max_tokens токенов
            max_length=self.max_tokens,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            h = self.model(**batch).last_hidden_state
            if self.pooling == "cls":
                v = h[:, 0]
            else:
                m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            v = torch.nn.functional.normalize(v.float(), dim=-1)
        return v.cpu().numpy()

    def vram_peak_mb(self) -> float:
        return self._torch.cuda.max_memory_allocated() / 2**20

    def reset_vram_peak(self) -> None:
        self._torch.cuda.reset_peak_memory_stats()


def embed_corpus(
    cfg,
    encoder: Encoder,
    model: str,
    max_tokens: int,
    cache_dir: str | Path,
    batch_size: int = 32,
    text_chars: int = 5000,
) -> list[Path]:
    """Кодирует train и test в обоих режимах шапки (`with`, `none`) и кэширует на диск.
    Тексты — голова документа (кэш усечённых текстов `text_chars` символов); меток не читается:
    берётся только колонка `text`. Возвращает пути четырёх файлов кэша."""
    from itgorod.data import build_t2_cache

    full = dataclasses.replace(cfg, max_chars=text_chars, tail_chars=0, strip_header=False)
    paths = []
    for split in ("train", "test"):
        texts = pd.read_parquet(build_t2_cache(full, split, smoke=False), columns=["text"])[
            "text"
        ].tolist()
        for header in ("with", "none"):
            spec = EmbeddingSpec(model, max_tokens, header, len(texts))
            encode_with_cache(encoder, texts_for_mode(texts, header), spec, cache_dir, batch_size)
            paths.append(cache_path(cache_dir, spec))
    return paths
