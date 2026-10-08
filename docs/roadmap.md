# Trust Layer: Roadmap (v0.1)

Phases are ordered so that something is always demoable. Hours are my own estimates, including learning time. There are no calendar dates on purpose: progress is tracked by exit criteria, not by the clock.

| # | Phase | Hours | Runs on |
|---|---|---|---|
| 1 | Foundations and repo setup | 5 | Mac |
| 2 | Data prep with PySpark | 8 | Mac |
| 3 | Labels and gold set | 6 | Mac |
| 4 | Embedding pipeline (text, image) | 10 | Mac + Colab |
| 5 | Permission-aware retrieval | 8 | Mac |
| 6 | Retrieval benchmark | 15 | Mac + Colab |
| 7 | Classifier comparison and calibration | 16 | Colab + Mac |
| 8 | Model service and C# gateway | 10 | Mac |
| 9 | Demo, deploy, README | 8 | Mac |
| 10 | Demo video | 6 | Mac |
| | **Core total** | **92** | |
| 11 | Stretch: audio | 16 | Mac + Colab |

## Working rules (every phase)

- The core logic (training loops, metrics, retrieval filter, Spark transforms) is written by me first; the AI agent reviews and suggests. The agent may scaffold boilerplate.
- Every number that appears in the README must trace to a run log or notebook.
- Check for NaN and wrong dimensions before trusting any embedding run.
- Each phase has a lesson in `docs/learnings/phaseN.md`: concepts, tips and tricks, spec pointers. Lessons are written alongside the work, not gated before or after it.

---

## Phase 1: Foundations and repo setup (5h)

**Goal:** a fresh clone builds, tests and starts the database with two commands.

- [x] Repo skeleton per `spec.md` section 14; licence; `.gitignore`; `.editorconfig`; `.env.example`
- [x] Python project (pinned versions), lint and test tooling, one passing placeholder test
- [x] .NET solution with a gateway project and a test project, one passing placeholder test
- [x] `docker-compose.yml` with Postgres + pgvector and a health check; migration folder
- [x] `Makefile` targets: `setup`, `test`, `lint`, `up`, `down`
- [x] GitHub Actions: build and test both stacks
- [x] `data/SOURCES.md` stub and `docs/learnings/` lesson convention
- [x] Verify Java version compatible with the pinned PySpark
- [x] Strict linting: ruff rules + format, .NET Recommended analyzers as errors, complexity ≤ 40 both stacks, pre-commit hook
- [x] `AGENTS.md` repository guidelines

**Exit criteria:** `make setup && make test` green locally and in CI; `make up` shows a healthy database.
**Lesson:** `docs/learnings/phase1.md`

## Phase 2: Data prep with PySpark (8h)

**Goal:** a cleaned, deduplicated, chunked Parquet corpus.

- [x] Download script (not data) for the dev subset (`data/scripts/download_aeslc.py`)
- [x] Parse emails, strip quoted replies and signatures where reasonable, extract thread IDs
- [x] Near-duplicate removal; dedup stats logged (`corpus-stats.json`)
- [x] Chunking with a documented size and overlap choice (1000/100, see lesson)
- [x] Output Parquet with a documented schema; unit tests on the transforms
- [x] Record row counts and runtime for the dev subset (18,302 → 23,267 chunks, 8.9s; full Enron run deferred — see below)

**Exit criteria:** reproducible Parquet from a single command (`make prep`); tests pass (17 Python + 1 xUnit).
**Lesson:** `docs/learnings/phase2.md`

**Follow-up (not blocking Phase 3):** full-corpus run on the Enron email corpus (Kaggle `wcukierski/enron-email-dataset` or CMU original) — needs the download script, a real `thread_id` from mail headers (replacing the pseudo-thread approximation), and recorded row counts/runtime at scale.

## Phase 3: Labels and gold set (6h)

**Goal:** trustworthy labels and a frozen test set.

- [x] Choose and document the labelled PII dataset; note licence and languages (`ai4privacy/pii-masking-43k`, custom free-for-small-team licence, English, synthetic — see SOURCES.md)
- [x] Define the label scheme (`public | internal | confidential`, plus PII flags) in `trustlayer.labels.rules`
- [x] Weak-label the email corpus with rules (16,650/5,791/826 over 23,267 chunks), then review 200 items as the gold set (agent-reviewed per your decision; agreement 123/200)
- [x] Freeze train, dev and test splits (by thread, 13,483/1,729/1,749 docs, seed 42); commit split manifests, not data
- [x] Write down known label limits in `docs/learnings/phase3.md`

**Exit criteria:** frozen gold set (`data/gold/gold.jsonl`, git-ignored) and split manifests (committed); labelling notes in the lesson.
**Lesson:** `docs/learnings/phase3.md`

## Phase 4: Embedding pipeline, text and image (10h)

**Goal:** resumable embedding of text and images with correct model usage.

- [x] Load text-only encoder; verify `mps` output against a CPU float32 reference (cosines ≥ 0.9999999)
- [x] Pin model revision (`914f7f8`) and record it in the lesson
- [x] Prefix handling (Document prompt via `encode_document`, no double-prefix), normalisation, dimension handling (768d)
- [x] Finite and dimension checks per batch; chunked output; resume after interruption (proven: local kill at 5/24, Colab redid all)
- [x] Text+image encoder path; image subset embedded (FUNSD 50, CPU batch, indexed `modality='image'`, cross-modal verified)
- [x] Thin Colab notebook that clones the repo and runs the module; saves to Drive per chunk (proven: 726s T4 run)
- [x] Time 1,000 documents and extrapolate before any full run (71.2s MPS → ~28 min predicted)
- [x] Load embeddings into pgvector (23,267 vectors, 15.3s; UPDATE...FROM after the upsert lesson)

**Exit criteria:** resume test passes; all vectors finite; dev subset fully indexed (16,961 text docs / 23,267 chunks + 50 image docs / 50 chunks, 0 nulls).
**Lesson:** `docs/learnings/phase4.md`

## Phase 5: Permission-aware retrieval (8h)

**Goal:** search that cannot leak.

- [x] Seed synthetic users, roles and ACLs (fixed seed — blake2b(doc_id), 11,573/4,724/714)
- [x] SQL pre-filter by `allowed_roles` and label rules (+ Python re-check, PermissionError on drift)
- [x] Test recall under filtering with HNSW; document behaviour and the fix (1 shortfall in 6,916 queries, `strict_order` ships in `search()`)
- [x] Leak test over every user and evaluation query (4 × 1,729, 69,160 hits, 0 violations)
- [x] Text and image results in one ranked list (modality per hit, mixed ranking verified)

**Exit criteria:** leak test shows zero violations (0/69,160, 96.5s).
**Lesson:** `docs/learnings/phase5.md`

## Phase 6: Retrieval benchmark (15h)

**Goal:** an auditable quality, memory and speed table.

- [x] Build and freeze the query set; hand-check at least 100 queries (300 frozen, sha `65390815`; 100 reviewed, 1 weak-but-valid)
- [x] Benchmark harness: recall@10, MRR@10, nDCG@10, latency, memory, index size (5 repeats, bootstrap CIs, config per run)
- [x] Baseline plus O1 to O7 and the silent-failure ablations (O4 blocked: llama.cpp lacks the arch — recorded)
- [x] Cross-lingual rows (Vietnamese and English) (EN→VI MRR 0.954 vs VI→VI 0.985; VI→EN needs parallel corpus — recorded)
- [x] Bootstrap confidence intervals; config logged per run
- [x] Results table, Pareto plot, "what I would ship" section (O7: README + lesson)

**Exit criteria:** README-ready results with intervals and limits stated (done — table, plot, limits).

## Phase 7: Classifier comparison and calibration (16h)

**Goal:** compare four ways of making a sensitivity decision, fairly.

- [ ] Logistic regression on frozen embeddings
- [ ] Fine-tuned DistilBERT on Colab
- [ ] LLM zero-shot baseline with cost and latency recorded
- [ ] MediaPipe Decision Maker as a zero-shot method; NaN check and float32 comparison; verify macOS support
- [ ] Same gold set, same splits for all; precision, recall, F1, confusion matrices
- [ ] Calibration: reliability diagram and expected calibration error
- [ ] 5 to 10 analysed errors

**Exit criteria:** one comparison table and a short written conclusion.
**Lesson:** `docs/learnings/phase7.md`

## Phase 8: Model service and C# gateway (10h)

**Goal:** the system works end to end behind a real API.

- [ ] FastAPI service: `/embed`, `/classify`, `/decide`, `/health`
- [ ] C# gateway: JWT demo auth, ACL enforcement, `/ask` (fast and answer modes), `/classify`, `/documents/{id}`
- [ ] Latency breakdown in responses; request IDs in logs
- [ ] xUnit tests for auth and ACL; API contract tests; compose-based end-to-end check

**Exit criteria:** contract tests green; leak test passes through the API.
**Lesson:** `docs/learnings/phase8.md`

## Phase 9: Demo, deploy, README (8h)

**Goal:** a public, skimmable repo with a live demo where possible.

- [ ] Gradio demo over the gateway
- [ ] Hosting, in order of preference: Hugging Face Spaces (free) with a precomputed small index; a Google option if credits allow; local Docker Compose always documented
- [ ] Architecture diagram; README per the voice guide in `spec.md`
- [ ] Limitations and "what I would do next" sections
- [ ] Final fresh-clone test

**Exit criteria:** acceptance criteria 1 to 5 and 7 from `spec.md` met.
**Lesson:** `docs/learnings/phase9.md`

## Phase 10: Demo video (6h)

**Goal:** a 90-second walkthrough.

- [ ] Script: ask a question, see a cited answer, see a sensitive item flagged, see a permission block
- [ ] Build with fframes in `demo-video/` (isolated). If setup exceeds about 2 hours, fall back to a screen recording
- [ ] Export screenshots or GIF frames for the README

**Exit criteria:** video embedded or linked in the README.
**Lesson:** `docs/learnings/phase10.md`

## Phase 11 (stretch, after everything above): audio (16h)

Only start when phases 1 to 10 are done. Mention in the README only if built and measured.

- [ ] FLEURS subset resampled to 16 kHz mono (English and Vietnamese)
- [ ] Audio encoder path and memory profile on the Mac
- [ ] Spoken or text query to audio retrieval; ground truth from paired transcripts
- [ ] Benchmark rows and a short write-up

**Lesson:** `docs/learnings/phase11.md`

---

## Cut list (if time runs short, cut in this order)

1. Phase 11 audio (already optional)
2. O6 half-precision vector storage
3. fframes video, replaced by a screen recording
4. Answer-mode LLM, keeping fast mode only
5. Public hosted demo, keeping a recorded demo and local instructions

**Never cut:** the leak test, the frozen gold set, confidence intervals, the limitations section, `docs/learnings/`.

## Definition of done

All acceptance criteria in `spec.md` section 13 are met, and I can explain every decision in the repo without notes.