# Phase 7 — Classifier comparison and calibration: lesson

Five rows on the same 200 gold items (public 64 / internal 119 /
confidential 17): LR on frozen Classification-prefix embeddings (± PII
synthetic), fine-tuned DistilBERT, Gemma 3 1B zero-shot, MediaPipe Decision
Maker (Laya). Train = 16,720 weak-labelled chunks on train-manifest docs
minus all 193 gold threads (spec §5 thread discipline).

## Comparison (primary, n=200)

| Method | Macro-F1 | P-con / R-con | ECE | p50 | Device |
|---|---|---|---|---|---|
| DistilBERT (3ep, cuda) | 0.598 | 0.27 / 1.00 | 0.37 | 6ms | T4 |
| LR 768cls (C=10) | 0.560 | 0.26 / 0.88 | 0.14 | ~0ms | cpu |
| LLM Gemma 3 1B zero-shot | 0.435 | 0.23 / 0.53 | 0.40 | 357ms | mps |
| Laya zero-shot | 0.393 | 0.29 / 0.12 | 0.10 | 80ms | cpu |
| LR + PII synthetic (ablated) | 0.394 | 0.22 / 0.71 | 0.37 | ~0ms | cpu |

Test-manifest secondary (n=20, wide CIs): DistilBERT 0.77 / LR 0.69 /
Laya 0.51 / PII 0.49 / LLM 0.37 — same ordering except the LLM row, which
is noise at n=20 (its CI spans 0.35–0.75 accuracy).

## Findings

- **Trained methods catch confidential; zero-shot (mostly) doesn't.**
  DistilBERT 17/17, LR 15/17, LLM 9/17, Laya 2/17. The corpus's
  confidential patterns (attorney-work-product phrasing, HR-strategy
  discussions, marker + payload combos) are learnable from weak labels
  but not guessable from label descriptions alone.
- **But confidential precision is terrible everywhere (0.22–0.29).**
  59 non-confidential rows get flagged by a trained method. The weak
  rules over-fire on the word "confidential" (footers, policy
  boilerplate, subject prefixes), and the trained models inherited the
  wolf-crying — rule-mimicry, exactly the Phase 3 prediction. A
  confidential flag from this system means "worth a human look", not
  "is confidential".
- **PII augmentation hurt (0.56 → 0.39 macro-F1).** The mapped synthetic
  set is 96% internal (41,194/42,760) — nearly every template carries a
  B- tag, so the mapping yields almost no public rows. Mixing it in
  drowned the real prior. Correct call keeping it ablated; logged as a
  negative result.
- **DistilBERT is badly miscalibrated (ECE 0.37)** despite the best F1:
  195/200 rows pile into the top confidence bin at 0.63 accuracy.
  Near-zero train loss (1.079 → 0.008) → overconfident. LR (ECE 0.14)
  and Laya (ECE 0.10) spread honestly. Best-F1 ≠ most-trustworthy-
  probabilities — the calibration section of the spec exists for this.
- **The LLM row is the interesting middle**: beats Laya on F1 and
  catches half the confidential items — a 1B instruct model reasons
  about sensitivity better than embedding-similarity to criteria. But
  worst-calibrated (sharp first-token probs, ECE 0.40) and 60× slower
  than DistilBERT inference. Its 12 "public" calls are 11/12 correct:
  it almost never says public unless sure (R-pub 0.17, P-pub 0.92).
- **Laya is honest but blind**: best ECE (0.10), worst confidential
  recall (2/17). Embedding-similarity to a 1-sentence criterion can't
  separate "mentions confidentiality" from "is confidential".
- **GLiNER float16 probe passes**: 20 rows, 0 errors, 0 NaN, 0 empty
  (agreement with Laya 10/20 — different model, expected). The spec §9
  risk (float16 build gives bad vectors) does not materialize on macOS;
  logged in `data/classify/gliner-probe.json`, never a comparison row.

## Confusion matrices (rows gold pub/int/con, cols pred pub/int/con)

- DistilBERT: [50,9,5] / [19,58,42] / [0,0,17] — never misses con, bleeds int→con.
- LR 768cls: [50,7,7] / [31,53,35] / [0,2,15] — same shape, slightly worse.
- LLM: [11,48,5] / [1,93,25] / [0,8,9] — public-phobic, internal-default.
- Laya: [23,39,2] / [35,81,3] / [5,10,2] — everything drifts internal.

## Calibration

Multiclass ECE over 10 equal-width max-prob bins; reliability data in
`data/classify/reliability.json`, plot in `docs/phase7-reliability.png`.
DistilBERT/LLM pile ~90% of rows into bin 9 (confidence ≈ 1.0, accuracy
≈ 0.6) — textbook overconfidence from low train loss / sharp token
probs. LR spreads across bins 4–9 with mild overconfidence. Laya tracks
the diagonal best. No temperature scaling applied (unprobed; the raw
numbers are the honest comparison).

## Error analysis (8 patterns, no email content quoted)

1. **Boilerplate-mimicry (both trained)**: privacy-policy and
   confidentiality-reminder boilerplate the gold review overruled to
   internal still gets flagged confidential — the models learned the
   rule's trigger (the word), not the reviewer's distinction (payload).
2. **Subject-prefix mimicry**: "Confidential:" subject on a routine body
   fools both trained methods; the review downgraded these.
3. **Generic-account phrasing**: "your account", course-title "Account
   Information" fire the account rule in training, and the models
   reproduce the false confidential.
4. **Attorney-word chatter**: standard-form legal drafting discussion
   (attorney word, no privilege) flagged confidential by both trained.
5. **Laya misses real confidential (15/17)**: attorney work product,
   layoff strategy, plaintext passwords all score below the
   "restricted content" criterion — similarity to a generic description
   is too weak a signal.
6. **LLM catches what Laya misses (9/17)** but misses subtler cases:
   HR do-not-share instructions and Chatham-House-style invitations go
   internal — the 1B model under-reads soft markers.
7. **LLM public-phobia**: only 12 public calls in 200 rows (11 correct).
   Trivial content (spam, banter, footers, cancellations) mostly lands
   internal — safe direction, but R-pub 0.17 tanks its macro-F1.
8. **The one LLM public miss**: a vendor report with "do not
   distribute" called public — redistribution-restriction vs
   sensitivity confusion. Genuinely ambiguous; the gold note debates it.

## Methodology notes

- **Thread-level gold exclusion**: train drops every doc sharing a gold
  doc's thread_id (344 docs / 2,144 chunks cut vs 162/543 doc-level;
  193 gold threads, max 32 docs). Spec §5 discipline held.
- **Eval = full 200 + test-20 secondary.** Strict split-discipline
  (test-gold only, n=20) is useless for 3-class F1 + ECE; the 200-row
  primary with thread-excluded train is the honest compromise, limits
  stated. 17 confidential positives → wide CIs on P-con/R-con.
- **C by 5-fold CV on weak train only** (no gold peeking): C=10 for
  both LR rows (CV macro-F1 0.60 / 0.76 — the PII row's higher CV score
  measures synthetic-mimicry, and gold exposed it).
- **Zero-shot purity**: Laya criteria and the LLM prompt were fixed
  a priori from rule tiers + spec ACL semantics, no tuning on gold.
  Wordings differ per API (Choice criteria vs chat prompt) but define
  the same three semantics — verified at integration.
- **Device per row** (spec §10): re-encode + DistilBERT-train on T4,
  everything else local (mps/cpu). Config in each results JSON.
- **Laya, not EmbeddingGemma**: no EG2 Decision Maker asset exists in
  Google's bucket (verified listing — only Laya + GLiNER). Laya float32
  is the row; GLiNER float16 was probed for NaN (spec risk check) and
  is not a comparison row.
- **Gemma 3 1B is text-only** (`gemma3_text`, `Gemma3ForCausalLM`) —
  `AutoModelForCausalLM` is the right class, verified against the
  pinned transformers mapping. Revision logs "unknown" (config lacks
  `_commit_hash`); model ID + date recorded instead.

## Tips and tricks

- `tail` exits 0 even when the piped command crashes — a `cmd | tail &&
  next` chain runs `next` after a failure. Check `${PIPESTATUS}` or run
  un-piped when chaining matters (a crashed LR run still launched its
  follow-up this way; caught by the missing output file).
- `--limit N` probe runs overwrite the results file — write probes to
  /tmp, not the default out path (restored by re-run; 15s, no harm).
- First-token-logprob softmax gives real zero-shot probabilities for
  ECE without a second forward pass — collect the label-token variants
  (bare/space/capitalized), max wins, renormalize.
- A `print` outside its `if` branch crashes only the no-flag path —
  the path you test second. Run both flag states before dispatching.

Spec pointers: §9, F2, F6.
