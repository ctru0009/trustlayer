# Data sources

Every public dataset this project touches gets a row here before it is used: exact version, licence
and the date I checked it. No data is committed — `data/` is git-ignored apart from this file and
`data/scripts/` (download scripts only).

The candidate sources are listed in section 5 of
[`../docs/spec.md`](../docs/spec.md) with the handling rules that apply to each.

| Source | Used for | Version pinned | Licence | Checked on |
|---|---|---|---|---|
| `Yale-LILY/aeslc` | Phase 2 dev corpus (Enron subject-line emails) | rev `2305f2e` (parquet: train 14,436 / validation 1,960 / test 1,906) | CC BY-NC-SA 4.0 (source repo `ryanzhumich/AESLC`; HF page lists `license:unknown`, paper arXiv:1906.03497) | 2026-10-08 |

## Rules I am holding myself to

- Verify the licence and availability at the moment of use; record it above rather than trusting
  what a tutorial said.
- The Enron corpus is real people's email. Scripts download it, nothing derived from it is
  published, and no individual message appears in a demo, screenshot or fixture.
- Synthetic users, roles and ACLs come from a seeded script so results are reproducible.
