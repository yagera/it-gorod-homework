# %% [markdown]
# # T2: замер скорости энкодеров на 1000 документов train (без меток, без полного кодирования)
# Запуск: `uv run python notebooks/05_t2_embedding_bench.py [--n 1000] [--max-tokens 512]`.
# Для каждого кандидата и режима шапки (`with` / `none`): документов в секунду, пик VRAM, оценка
# времени на ~50 тыс. документов. Тексты — первые N документов train из кэша усечённых текстов
# (5000 символов, голова документа); метки не читаются. Результат — `outputs/embedding_bench.csv`.

# %%
import argparse
import dataclasses
import gc
import time

import pandas as pd
import torch

from itgorod.config import REPO_ROOT, load_t2_config
from itgorod.data import build_t2_cache
from itgorod.embeddings import CANDIDATES, HFEncoder, benchmark_encoder, texts_for_mode

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--models", nargs="*", default=list(CANDIDATES))
args = ap.parse_args()

cfg = load_t2_config(REPO_ROOT / "configs" / "t2.yaml")
cache = build_t2_cache(dataclasses.replace(cfg, max_chars=5000), "train", smoke=False)
texts = pd.read_parquet(cache, columns=["text"])["text"].head(args.n).tolist()  # только текст
print(f"документов: {len(texts)}, GPU: {torch.cuda.get_device_name(0)}")

# %%
rows = []
for model in args.models:
    t0 = time.time()
    enc = HFEncoder(model, max_tokens=args.max_tokens)
    load_s = time.time() - t0
    enc.encode(texts[:8])  # прогрев (не входит в замер)
    for header in ("with", "none"):
        batch_texts = texts_for_mode(texts, header)
        enc.reset_vram_peak()
        torch.cuda.synchronize()
        res = benchmark_encoder(enc, batch_texts, args.batch_size, vram_peak_mb=enc.vram_peak_mb)
        torch.cuda.synchronize()
        res |= {
            "model": model,
            "header": header,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "load_s": round(load_s, 1),
            "params_m": CANDIDATES[model]["params_m"],
        }
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
        rows.append(res)
    del enc
    gc.collect()
    torch.cuda.empty_cache()

out = REPO_ROOT / "outputs" / "embedding_bench.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("записано:", out)
