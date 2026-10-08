# Data sources

Every public dataset this project touches gets a row here before it is used: exact version, licence
and the date I checked it. No data is committed — `data/` is git-ignored apart from this file and
`data/scripts/` (download scripts only).

The candidate sources are listed in section 5 of
[`../docs/spec.md`](../docs/spec.md) with the handling rules that apply to each.

| Source | Used for | Version pinned | Licence | Checked on |
|---|---|---|---|---|
| `Yale-LILY/aeslc` | Phase 2 dev corpus (Enron subject-line emails) | rev `2305f2e` (parquet: train 14,436 / validation 1,960 / test 1,906) | `license:unknown` in HF metadata; source repo `ryanzhumich/AESLC` claims CC BY-NC-SA 4.0 — HF page is unverified, NC clause assumed binding for any hosted demo (paper arXiv:1906.03497) | 2026-10-08 |
| `ai4privacy/pii-masking-43k` | Phase 3 PII ground truth / Phase 7 classifier training (English, labelled, synthetic token-classification) | rev `7d38093` (`PII43k.csv`, 27.7 MB, 42,760 data rows; 59 stray NUL bytes after file line 42759, strip before CSV parse) | No `license` tag in HF API metadata; repo `LICENCE` file (verified at pinned rev) is a custom free-for-individuals/small-business (≤3 staff) licence, corporate licence required above that — OK for this learning project, re-check before any hosted demo. (Flagship `pii-masking-openpii-1m` is `license:other` + `license_name:cc-by-4.0` with README CC-BY-4.0 grant, but single multilingual 4.6 GB file train+validation, English not separable — deferred.) | 2026-10-08 |

## Rules I am holding myself to

- Verify the licence and availability at the moment of use; record it above rather than trusting
  what a tutorial said.
- The Enron corpus is real people's email. Scripts download it, nothing derived from it is
  published, and no individual message appears in a demo, screenshot or fixture.
- Synthetic users, roles and ACLs come from a seeded script so results are reproducible.
