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

- Concept first, code second: write 3 to 5 lines in `LEARNING.md` before starting a phase, and 3 to 5 more after.
- The core logic (training loops, metrics, retrieval filter, Spark transforms) is written by me first; the AI agent reviews and suggests. The agent may scaffold boilerplate.
- After each phase, get quizzed (5 questions, one at a time) and fix the gaps.
- Every number that appears in the README must trace to a run log or notebook.
- Check for NaN and wrong dimensions before trusting any embedding run.

---

## Phase 1: Foundations and repo setup (5h)

**Goal:** a fresh clone builds, tests and starts the database with two commands.

- [ ] Repo skeleton per `spec.md` section 14; licence; `.gitignore`; `.editorconfig`; `.env.example`
- [ ] Python project (pinned versions), lint and test tooling, one passing placeholder test
- [ ] .NET solution with a gateway project and a test project, one passing placeholder test
- [ ] `docker-compose.yml` with Postgres + pgvector and a health check; migration folder
- [ ] `Makefile` targets: `setup`, `test`, `lint`, `up`, `down`
- [ ] GitHub Actions: build and test both stacks
- [ ] `data/SOURCES.md` and `LEARNING.md` stubs
- [ ] Verify Java version compatible with the pinned PySpark

**Exit criteria:** `make setup && make test` green locally and in CI; `make up` shows a healthy database.
**Checkpoint:** why pin versions? What does a health check give you? What would break on a different machine?

## Phase 2: Data prep with PySpark (8h)

**Goal:** a cleaned, deduplicated, chunked Parquet corpus.

- [ ] Download scripts (not data) for the dev subset, then the full corpus
- [ ] Parse emails, strip quoted replies and signatures where reasonable, extract thread IDs
- [ ] Near-duplicate removal; dedup stats logged
- [ ] Chunking with a documented size and overlap choice
- [ ] Output Parquet with a documented schema; unit tests on the transforms
- [ ] Record row counts and runtime for the dev subset and the full run

**Exit criteria:** reproducible Parquet from a single command; tests pass.
**Checkpoint:** what does a Spark shuffle cost? Why dedupe before splitting? Why split by thread?

## Phase 3: Labels and gold set (6h)

**Goal:** trustworthy labels and a frozen test set.

- [ ] Choose and document the labelled PII dataset; note licence and languages
- [ ] Define the label scheme (`public | internal | confidential`, plus PII flags)
- [ ] Weak-label the email corpus with rules, then hand-check 200 items as the gold set
- [ ] Freeze train, dev and test splits (by thread); commit split manifests, not data
- [ ] Write down known label limits in `LEARNING.md`

**Exit criteria:** frozen gold set and split manifests; labelling notes written.
**Checkpoint:** what is label noise and how does it inflate scores? Why precision and recall over accuracy here?

## Phase 4: Embedding pipeline, text and image (10h)

**Goal:** resumable embedding of text and images with correct model usage.

- [ ] Load text-only encoder; verify `mps` output against a CPU float32 reference
- [ ] Prefix handling (`SearchQuery`, `Document`, `Classification`), normalisation, dimension handling
- [ ] Finite and dimension checks per batch; chunked output; resume after interruption
- [ ] Text+image encoder path; image subset embedded
- [ ] Thin Colab notebook that clones the repo and runs the module; saves to Drive per chunk
- [ ] Time 1,000 documents and extrapolate before any full run
- [ ] Load embeddings into pgvector

**Exit criteria:** resume test passes; all vectors finite; dev subset fully indexed.
**Checkpoint:** why not float16? What is an embedding space? Why do prefixes matter?

## Phase 5: Permission-aware retrieval (8h)

**Goal:** search that cannot leak.

- [ ] Seed synthetic users, roles and ACLs (fixed seed)
- [ ] SQL pre-filter by `allowed_roles` and label rules
- [ ] Test recall under filtering with HNSW; document behaviour and the fix
- [ ] Leak test over every user and evaluation query
- [ ] Text and image results in one ranked list

**Exit criteria:** leak test shows zero violations.
**Checkpoint:** why filter before ranking? What does approximate search trade away?

## Phase 6: Retrieval benchmark (15h)

**Goal:** an auditable quality, memory and speed table.

- [ ] Build and freeze the query set; hand-check at least 100 queries
- [ ] Benchmark harness: recall@10, MRR@10, nDCG@10, latency, memory, index size
- [ ] Baseline plus O1 to O7 and the silent-failure ablations
- [ ] Cross-lingual rows (Vietnamese and English)
- [ ] Bootstrap confidence intervals; config logged per run
- [ ] Results table, Pareto plot, "what I would ship" section

**Exit criteria:** README-ready results with intervals and limits stated.
**Checkpoint:** recall@k vs MRR vs nDCG; why does 128d hurt images more than text; what would change your recommendation at 100x scale?

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
**Checkpoint:** what is data leakage; what does calibration mean; why F1 over accuracy on imbalanced labels?

## Phase 8: Model service and C# gateway (10h)

**Goal:** the system works end to end behind a real API.

- [ ] FastAPI service: `/embed`, `/classify`, `/decide`, `/health`
- [ ] C# gateway: JWT demo auth, ACL enforcement, `/ask` (fast and answer modes), `/classify`, `/documents/{id}`
- [ ] Latency breakdown in responses; request IDs in logs
- [ ] xUnit tests for auth and ACL; API contract tests; compose-based end-to-end check

**Exit criteria:** contract tests green; leak test passes through the API.
**Checkpoint:** why keep permissions out of the model service? What happens if the model service is down?

## Phase 9: Demo, deploy, README (8h)

**Goal:** a public, skimmable repo with a live demo where possible.

- [ ] Gradio demo over the gateway
- [ ] Hosting, in order of preference: Hugging Face Spaces (free) with a precomputed small index; a Google option if credits allow; local Docker Compose always documented
- [ ] Architecture diagram; README per the voice guide in `spec.md`
- [ ] Limitations and "what I would do next" sections
- [ ] Final fresh-clone test

**Exit criteria:** acceptance criteria 1 to 5 and 7 from `spec.md` met.

## Phase 10: Demo video (6h)

**Goal:** a 90-second walkthrough.

- [ ] Script: ask a question, see a cited answer, see a sensitive item flagged, see a permission block
- [ ] Build with fframes in `demo-video/` (isolated). If setup exceeds about 2 hours, fall back to a screen recording
- [ ] Export screenshots or GIF frames for the README

**Exit criteria:** video embedded or linked in the README.

## Phase 11 (stretch, after everything above): audio (16h)

Only start when phases 1 to 10 are done. Mention in the README only if built and measured.

- [ ] FLEURS subset resampled to 16 kHz mono (English and Vietnamese)
- [ ] Audio encoder path and memory profile on the Mac
- [ ] Spoken or text query to audio retrieval; ground truth from paired transcripts
- [ ] Benchmark rows and a short write-up

---

## Cut list (if time runs short, cut in this order)

1. Phase 11 audio (already optional)
2. O6 half-precision vector storage
3. fframes video, replaced by a screen recording
4. Answer-mode LLM, keeping fast mode only
5. Public hosted demo, keeping a recorded demo and local instructions

**Never cut:** the leak test, the frozen gold set, confidence intervals, the limitations section, `LEARNING.md`.

## Definition of done

All acceptance criteria in `spec.md` section 13 are met, and I can explain every decision in the repo without notes.