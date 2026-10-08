# Phase 7 Colab guide: re-encode + DistilBERT on a T4

Two thin notebooks do the GPU-heavy Phase 7 work (~25–35 min total, mostly
unattended). Everything else runs locally. Do the notebooks in either order —
they are independent of each other.

## What runs where (corrected)

| Job | Where | Why |
|---|---|---|
| Re-encode (23k Classification vectors) | Colab T4 (`phase7a`) | ~12 min vs 1h+ local |
| DistilBERT train + predict | Colab T4 (`phase7b`) | ~10–20 min vs 30–90 min MPS |
| Decision Maker 200 rows | Local (me, now) | CPU task, no GPU gain |
| LR train + eval | Local (me, after 7a artifacts land) | Trains ON the re-encoded vectors — must wait for 7a |
| Gemma 3 1B 200 rows | Local (me, needs your HF token) | 1B fits in 16 GB; Enron stays on-laptop |

Device is logged per result row (spec §10: never mix laptop/Colab numbers in
one row — here each method is its own row, device recorded in its config).

## Notebook 1: re-encode (`notebooks/phase7a_vectors.ipynb`)

1. Upload the notebook to Colab (File → Upload notebook).
2. Runtime → Change runtime type → **T4 GPU**.
3. Run all cells in order. Cell 1 mounts Drive + clones + installs (~3 min);
   cell 2 installs Java; cell 3 downloads AESLC + builds the corpus (~2 min);
   cell 4 encodes 23,267 chunks and saves each chunk file to
   `Drive/trustlayer/embeddings-cls/` as it lands (~10–15 min).
4. Watch for the prompt line near the top of cell 4's output:
   `prompt 'Classification' -> 'task: classification | ...'`. That string is
   the Colab-side proof the right prompt applied (the module raises if the
   prompt is missing). The cls-vs-Document cosine check runs locally after
   download (step 6) because rebuilding the Phase 4 Document vectors on Colab
   would cost another ~12 min for no reason.
5. If Colab disconnects: re-run — completed chunks restore from Drive and are
   skipped (resume). Nothing is lost.
6. When cell 4 prints `exit: 0`, download from Drive back to the repo:
   - `trustlayer/embeddings-cls/chunk-*.jsonl` (24 files, ~70 MB) →
     `data/processed/embeddings-cls/` (8 chunks already exist locally from the
     cancelled run — same format, keep them; only missing ones matter, but
     re-downloading all 24 is simplest and harmless).
   - `trustlayer/embeddings-cls/embeddings-cls-stats.json` →
     `data/processed/embeddings-cls-stats.json`.
7. Tell me it's landed — I run the sanity check (cls-vs-Document cosine on
   100 rows, must be < 0.99) and then LR training locally.

## Notebook 2: DistilBERT (`notebooks/phase7b_distilbert.ipynb`)

1. Upload to Colab, T4 GPU runtime, run cells in order.
2. Cell 2 opens a file picker — upload from your local repo:
   - `data/classify/train.jsonl` (~11 MB)
   - `data/classify/eval.jsonl` (~0.2 MB)
3. Cell 3 runs smoke (50 steps — check the loss drops) → 3-epoch train →
   200-row predict (~10–20 min). Let it finish; training does not resume.
4. Cell 4 backs up to `Drive/trustlayer/distilbert/`. Download:
   - `distilbert.json` (~100 KB) → `data/classify/results/distilbert.json`
     (this is the comparison row).
   - `model/` (~260 MB, optional) → `data/classify/distilbert/` (only if you
     want local re-predict; the predictions file is what the table needs).
5. Tell me it's landed — I fold it into the comparison.

## Troubleshooting

- **No T4 available**: any GPU runtime works (times shift a bit); CPU-only
  runtime defeats the purpose — wait for a GPU.
- **Drive out of space**: the two backups total ~330 MB; free space or skip
  the optional `model/` backup (predictions are the essential artifact).
- **`uv sync` slow**: first install downloads torch (~800 MB); ~3 min is normal.
- **Wrong chunk count** (not 24 files): check cell 4's `rows=` line says 23267;
  fewer means the corpus build (cell 3) used a partial download — re-run cell 3.
