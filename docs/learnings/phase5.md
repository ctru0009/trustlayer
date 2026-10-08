# Phase 5 — Permission-aware retrieval: lesson

Search that cannot leak. The permission predicate lives in the SQL
itself (`WHERE allowed_roles && roles`), and every returned row is
re-checked in Python before returning — defence in depth per spec §8.4.

## The predicate (spec §8.1)

Users: alice (HR), bob (Finance), carol (Everyone), admin. A user may
see a document iff a role is shared between user and grant; confidential
additionally requires a non-Everyone role in common — the explicit-grant
rule. `trustlayer/retrieve/acl.py:visible` is the single definition;
`search.py` implements it twice (SQL pre-filter + Python re-check) and
raises `PermissionError` on any disagreement rather than leaking.

SQL and Python must stay identical. The confidential clause is the
subtle half: `d.allowed_roles && nonEveryoneRoles` — carol's list is
empty (`[]`), so `&& '{}'` is false and she can never match a
confidential row, whatever the grant says.

## Seeded ACLs

`seed.py:assign` is pure and deterministic (blake2b of doc_id picks the
team; no RNG state to persist). Rules: public → {Everyone}; internal →
{Everyone, team}; confidential → {team} only; FUNSD images → {Everyone}
(public test fixtures, recorded choice). Distribution on the dev index:
11,523 public/Everyone, 4,724 internal split 2,327/2,397 across teams,
714 confidential split 343/371, 50 images. Confidential rows never
carry Everyone — verified by GROUP BY, not by trust.

## HNSW under filtering (spec §8.4.4)

Migration 002 builds the HNSW index (m=16, ef_construction=64). The
spec's warning — filtered HNSW can return fewer than LIMIT rows —
was probed, not assumed:

- Common filters (carol 96% visible, alice 98%): 10/10 rows on all 10
  probe queries, 4/100 misses vs exact (normal approximation error).
  HNSW ~0.5ms vs exact ~3.5ms per query.
- Selective filter (confidential-HR only, 2% of corpus): 10/10 rows,
  0/100 misses — no starvation at 23k rows.

No shortfall observed at this scale. If it appears at larger scale,
the documented fix is `hnsw.iterative_scan = 'strict_order'` — an
enum in pgvector 0.8.7 (`off | relaxed_order | strict_order`), NOT a
boolean (`=on` errors). Lesson learned by trying `=on` first.

Separate concern, kept separate: the leak test asserts permission
correctness, never recall. Approximate search may miss rows but must
never return a forbidden one.

## Leak test

`make leak`: every user × every dev query (1,729 first-chunk texts,
SearchQuery prompt, top-10) with an independent audit of every hit
against `acl.visible` — not trusting `search()`'s internal re-check.

```
{"users": 4, "queries": 1729, "hits_checked": 69157, "violations": 0,
 "seconds": 94.9}
```

69,157 hits independently audited, zero violations. Exit criteria met.

## Text + image in one list (F9)

`search()` joins chunks→documents and returns `modality` per hit; the
caller makes no distinction. Verified: an invoice query over the mixed
index returns text hits at dist ~0.25 and FUNSD forms at ~0.33 in one
ranking (Phase 4 numbers, same index).

## Design decisions and why

- **Predicate in one pure function** (`acl.visible`): SQL mirrors it,
  Python re-checks it, tests pin it. Three expressions of one rule —
  drift shows up as a PermissionError, not a leak.
- **Deterministic seed, no RNG**: blake2b(doc_id) assigns teams.
  Re-running `make seed-acls` converges to the same ACLs; no seed
  file to version.
- **`PermissionError`, not assert**: the re-check is a security
  boundary, and `python -O` strips asserts. Ruff S101 enforces this.
- **DB tests skip without DATABASE_URL**: pure-ACL tests run in CI;
  search/leak tests need the index and run locally via make targets.

## Known limits

- HNSW no-shortfall result is specific to 23k rows; larger corpora or
  tighter filters need re-probing (the probe script pattern is above).
- Leak queries are first-chunk truncations, not natural questions —
  they cover the permission surface, not retrieval quality (Phase 6).
- No C# gateway yet (Phase 8): the predicate will move there, with
  this Python path as the reference implementation.

## Tips and tricks

- pgvector GUCs (`hnsw.*`) don't exist until the library loads in the
  session — run any `<=>` query first or `SHOW` errors misleadingly.
- `SET` takes no bind parameters; interpolate on/off literals.
- `DISTINCT ON (doc_id) ... ORDER BY doc_id, ord` gives one query per
  doc in a single round trip.

Spec pointers: §8, F7, F9.
