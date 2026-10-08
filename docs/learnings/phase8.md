# Phase 8 — Model service and C# gateway: lesson

The system works end to end behind a real API: FastAPI model service
(`/embed`, `/classify`, `/decide`, `/answer`, `/health`) + C# gateway
(JWT demo auth, ACL enforcement, `/ask` fast/answer, `/classify`,
`/documents/{id}`). API leak test: 840/840 hits, 0 violations.

## What was built

- **Model service** (`python/src/trustlayer/service/`): thin FastAPI over
  the Phase 4/7 model code. Owns model loading + prefixes; knows nothing
  about users or permissions (spec §4 must-not holds — no user field in
  the module). Lazy holders: startup is instant, `/health` reports
  per-model readiness.
- **LR weights as a committed artifact**: `lr.py --export-weights` writes
  coef/intercept/classes (~47KB JSON) next to the service; pure-Python
  softmax inference (no sklearn at serve time). The refactored training
  path reproduces Phase 7 exactly (200/200 preds, max prob Δ = 0.0),
  and the service test replays 8 gold rows through the artifact.
- **Gateway** (`dotnet/src/TrustLayer.Gateway/`): minimal API. Retrieval
  SQL is an exact port of `retrieve/search.py` (pre-filter +
  `strict_order` + re-check; drift throws `AclDriftException`, never
  silently drops). JWT demo login (HMAC-SHA256, 8h, no passwords per
  spec). Missing-or-invisible → identical 404 everywhere.
- **Answer mode**: gateway passes permitted passages (cap 1500 chars
  each) to the service, which prompts Gemma 3 1B with a cite-or-decline
  instruction. Citations align 1:1 with returned hits (the leak test
  asserts `cited ⊆ returned`).
- **Compose**: `db` stays the default profile; `stack` adds
  service + gateway (`make stack`). Service image carries torch CPU;
  model/HF caches persist in a named volume.

## Findings

- **API leak test passes (0/840)**: 20 dev queries × 4 users fast mode
  + 2 × 4 answer mode, every hit audited against `acl.visible` ground
  truth from the DB, plus anti-vacuous (full top_k every query),
  no-answer-in-fast-mode, and citation-drift checks. 95s wall.
- **Cold-start latency is the honestly-measured number**: first `/ask`
  pays ~10s (encoder load) and answer mode ~6s (Gemma load + generate);
  the latency breakdown in the response shows where. No warm-up hiding.
- **Laya must be a session, not a call**: the first `/decide` cut opened
  the 678MB maker per request (2.8s wall). `LayaSession` holds one
  maker + question open → 71ms evaluate-only on call 2.
- **Train-serve skew is one line**: training formats
  `f"{title.strip() or 'none'}\n{text}"`; the service initially sent
  `f"{title}\n{text}"` — empty titles diverge. Now byte-identical.
- **Dimension is pinned, not negotiated**: spec §8.2 accepts
  `dimension`, but `chunks.embedding` is `vector(768)` and spec §6
  requires query/doc dims match — the gateway 400s anything but 768.
  Wiring Phase 6 variant tables is future work, not silently garbage.
- **CI does not build the stack** (deliberate): the torch service image
  (~2GB download) on every push is unjustified; e2e is a local
  `make stack` + `make api-leak` step. CI still runs both unit suites
  + the fresh-clone smoke.

## Methodology notes

- **Gateway tests are three layers**: ACL truth table (11 cases incl.
  Everyone-never-confidential), WebApplicationFactory contracts with an
  in-memory store + stubbed model client (auth 401s, ACL filtering
  through the API, 404-identical, validation 400s), and live-Postgres
  integration (pre-filter/re-check agreement) gated on
  `TRUSTLAYER_TEST_DB`.
- **xUnit 2.9 has no `Assert.Skip`** (that's xUnit.v3): the gate is a
  `FactAttribute` subclass setting `Skip` at discovery. A prior
  DLL-strings "confirmation" of `Assert.Skip` was a false positive
  (the string was a v3-compat resource name) — verify APIs by
  compiling, not by grepping binaries.
- **Wire format is snake_case** (spec §8.2 JSON): `top_k`, `doc_id`,
  `latency_ms` need `[JsonPropertyName]` — System.Text.Json binds
  case-insensitively but underscore-sensitively, so `top_k` silently
  missed `TopK` until 5 contract tests caught it.
- **Score = 1 − cosine distance** on normalized vectors (0..2 → 1..−1,
  in practice 0..1): similarity so higher ranks first. Documented in
  code; the contract test asserts the range.
- **`dotnet run` ignores `ASPNETCORE_URLS`** when launchSettings.json
  exists — pass `--urls` explicitly for local multi-service runs.

## Tips and tricks

- Multi-PUT edits to one file in a single call use pre-edit line
  numbers for every PUT — a second PUT in the same call easily lands
  on shifted lines. One PUT per call for same-file edits, or re-read
  between. (Bit twice this phase; the fix both times was rewriting
  the damaged span from a fresh read.)
- `edit` range endpoints inside a multi-line `def`/`(` span silently
  eat the opener — anchor insertions to blank lines between blocks,
  never to a `def` line.
- `docker compose build.context` resolves against the project
  directory, not the compose file's directory: with `-f
  infra/docker-compose.yml`, `context: ..` escapes the repo. Use
  `context: .` + `dockerfile: infra/...`.
- `from package import name` binds the SUBMODULE when one exists —
  a lazy `__getattr__` export with the same name as a submodule never
  fires. Import from the submodule directly.
- `TestClient` deprecation warning (`httpx` vs `httpx2`) is upstream
  starlette noise in the pinned versions — ignored, not worked around.
- A delegated gateway build stalled on research; pulling it back and
  building directly was faster than steering. Delegate bounded,
  verifiable slices — not "build the subsystem".

Spec pointers: §4, §8, F5, F6, F7.
