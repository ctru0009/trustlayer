# Phase 3 — Labels and gold set: lesson

Weak-labeller (`trustlayer.labels.rules`) + 200-item reviewed gold set +
frozen per-thread split manifests. One command reproduces the mechanical
parts: `make labels` (weak labels, gold sample, manifests); the gold review
is recorded judgement, not re-runnable code.

## Run results (2026-10-08, Mac, 6.9s)

Weak labels over 23,267 chunks: public 16,650 / internal 5,791 /
confidential 826. Gold sample stratified 68/66/66 across weak labels;
reviewed gold distribution: public 64 / internal 119 / confidential 17.
Weak→gold agreement 123/200 (61.5%).

Splits (by thread hash, seed 42, 80/10/10): train 13,483 / dev 1,729 /
test 1,749 docs. Verified: full 16,961-doc coverage, zero doc overlap,
zero thread overlap across splits.

## Label scheme and why

`public | internal | confidential` (spec §8.1) plus coarse PII flags
(email, phone, ssn, card, passport, credentials, account). Two severity
tiers: high-severity PII (ssn, card, passport, credentials, account) or an
explicit confidentiality marker → confidential; anything else with PII or
internal-business keywords → internal; otherwise public.

Rules fire on subject + body jointly — a "Privileged and Confidential"
subject with a trivial body still counts the marker, which the gold review
later corrects when the body has no substance. Card numbers must pass Luhn:
without the checksum, any 16-digit run (ticket numbers, reference ids)
would be a false positive; the gold set contains exactly such a case.

## What label noise is, and how it inflates scores

38.5% of weak labels disagreed with review. If those weak labels were used
as ground truth for Phase 7, a classifier could score highly by learning
the rules' mistakes (e.g. "boilerplate footer ⇒ confidential") rather than
real sensitivity — accuracy on noisy labels measures rule-mimicry, not
understanding. The gold set exists so at least 200 items measure the real
thing; every overrule below is a documented instance of noise removed.

## Why precision and recall over accuracy here

Confidential is 17/200 gold (8.5%) and 826/23,267 weak (3.5%). A classifier
that never predicts confidential scores 91.5% accuracy on gold while being
useless — accuracy is dominated by the majority class. Precision ("of items
flagged confidential, how many are?") and recall ("of truly confidential
items, how many did we catch?") measure the minority class directly, which
is the class whose misses cause leaks.

## Gold review patterns (no email content quoted)

Systematic weak-label failure modes found during review:

- **Boilerplate footers**: standard confidentiality footers on trivial
  content (social mail, newsletters) → weak confidential, gold public.
  The rule cannot distinguish footer from substance.
- **Boilerplate policies**: confidentiality-reminder and privacy-policy
  texts describe confidentiality without containing secrets → weak
  confidential, gold internal.
- **Generic "account" phrasing**: "bring the account current", "Account
  Manager" (job title), "take into account" all fired the account rule →
  overruled to internal/public. The rule needs a number nearby.
- **Attorney-word matches**: job titles in signatures, "City Attorney" in
  procurement notices, public court filings → weak confidential, gold
  internal. The word is not the privilege.
- **Subject-only markers**: confidential subject line with a substance-free
  body → downgraded; sensitivity needs payload, not just a flag.
- **Missed PII (upgrades)**: non-standard phone formats, home numbers in
  prose, payment-adjacent requests ("give her an account number") →
  upgraded public→internal. Rules under-match as well as over-match.
- **Spam/scams**: a 419 solicitation using "confidential" as a lure →
  internal, not confidential; the marker serves the attacker, not the owner.

Net effect of review: confidential shrank 66→17 (rules over-fire on the
word "confidential"), internal grew 66→119 (business content the
keyword list partially missed), public roughly held 68→64.

## Known label limits

- **Agent-reviewed, not human-reviewed.** You chose agent review of all 200
  (no human write override extended to labelling). Every downstream metric
  tracing to this gold set measures agreement with agent judgement. A human
  re-review of the 77 overrules (or a sample) is the single highest-value
  validation available later.
- **Pseudo-threads bound split purity.** Splits are by subject-hash thread,
  an approximation (Phase 2 lesson). Same-subject-different-topic emails
  group together (harmless: they're still separated jointly); genuinely
  related emails with different subjects can straddle splits (the real leak
  direction). Full-corpus mail headers fix this.
- **Chunk-level labels, doc-level splits.** A doc's chunks can carry mixed
  weak labels while the doc sits in one split — fine for retrieval eval,
  but Phase 7 training must decide per-chunk vs per-doc supervision.
- **PII flags are coarse.** The rules detect PII presence, not spans; span
  ground truth comes from the ai4privacy PII-43k dataset (SOURCES.md),
  which is synthetic and English-only — a domain gap from real email.
- **PII-43k licence is custom** (free for individuals/small teams, corporate
  licence above) — re-check before any hosted demo. The CC-BY-4.0 flagship
  (openpii-1m) was deferred: 4.6 GB single multilingual file, English not
  separable.
- **Enron handling**: gold.jsonl embeds chunk text and stays git-ignored;
  only doc_id manifests (content hashes) are committed. No individual
  message appears in docs, fixtures, or commits — this lesson describes
  patterns only.

## Tips and tricks

- Stratify the gold sample by weak label (thirds here), not uniformly:
  uniform sampling would yield ~7 confidential items and teach nothing
  about the minority class.
- Join verdicts to candidates by script (`(doc_id, chunk_ord)` key) and
  assert 200/200 coverage with zero empty notes — review output is data,
  validate it like data.
- Verify split manifests independently of the writer: coverage, zero doc
  overlap, zero thread overlap — three queries, each catching a different
  freeze bug.
- The `.gitignore` parent chain (`!data/gold/`, `data/gold/*`,
  `!data/gold/splits/`, ...) is required because git cannot re-include
  under an excluded directory; verified with `git check-ignore` both
  directions (manifests committable, gold content ignored).

Spec pointers: §5, §8, F2.
