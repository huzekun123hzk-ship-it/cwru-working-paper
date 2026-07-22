# Prompt Log

Chronological, auditable record of AI-assisted steps, following
`Final_Experiment/PROMPT_PROTOCOL.md`. Each entry records: date, tool/model, objective,
files/sources supplied, prompt summary, output used, verification performed, and errors
found. The AI ran locally; every number in the paper comes from committed scripts and
saved artifacts (`results/tables/*.json`), not from model text.

Tool/model used throughout: Cursor agent (Claude, `claude-opus-4.8`), running locally
against this repository. Dates are the working dates in July 2026.

---

## Entry 1 — Planning and direction
- Date: 2026-07 (planning phase)
- Tool/model: Cursor agent (Claude)
- Objective: pick a defensible, innovative research direction and define deliverables.
- Files/sources supplied: `CAPSTONE_GUIDE.md`, `Final_Experiment/manual.md`,
  `innovation_hints_cn.md`, prior small-assignment code under `../cv2/2025E`.
- Prompt summary: "Read the project docs and my prior work; propose the strongest,
  most innovative direction with a concrete deliverable list. Separate facts from
  suggestions; flag leakage/validity risks."
- Output used: chose combined Direction 1 (leakage-safe evaluation) + Direction 5
  (Grad-CAM physical consistency); defined the Interpretability Leakage Gap (ILG) as the
  named contribution.
- Verification performed: cross-checked deliverable list against the capstone guide's
  assessment criteria; confirmed the direction was not just an accuracy race.
- Errors found: none at this stage.

## Entry 2 — Repo scaffolding and data pipeline (coding)
- Date: 2026-07 (build phase)
- Tool/model: Cursor agent (Claude)
- Objective: build a reproducible pipeline (manifest -> windows -> envelope -> CNN).
- Files/sources supplied: `references/file_metadata.json`, `references/bearing_params.json`,
  raw CWRU `.mat` files (reused from prior coursework).
- Prompt summary: "Implement one module at a time with sanity checks. Build a manifest
  from the audited metadata, cut 2048/1024 windows, implement protocols R (random) and S
  (recording-level). Do not invent results."
- Output used: `src/` modules (config, io_cwru, preprocess, make_splits, ...); 7533 DE
  windows; installed torch (CPU) + scikit-learn.
- Verification performed: printed split overlap counts (recordings shared between
  train/test): R = 62 (fully overlapping by design), S = 0 (disjoint). Asserted class
  stratification.
- Errors found: two `.mat` files unreadable (`Normal_3HP_1730rpm.mat`,
  `IR007_1HP_1772rpm.mat`) -> added try/except in `io_cwru.load_de_signal`, skipped with
  a warning (62 of 64 recordings usable).

## Entry 3 — Diagnostics that changed the design (honesty trail)
- Date: 2026-07 (build phase)
- Tool/model: Cursor agent (Claude)
- Objective: check whether the "leakage inflates accuracy" framing actually holds.
- Files/sources supplied: pipeline outputs; `diag_difficulty.py`, `diag_hard.py`,
  `diag_saliency.py`, `diag_envelope_pc.py`.
- Prompt summary: "Before writing anything, empirically test the hypothesis. If it
  fails, tell me and refine — do not write around a failed experiment."
- Output used and findings:
  - DIAG 1: full-data accuracy saturates for both R and S (~100%). Original framing too
    weak on full data -> refined thesis to "accuracy is blind, interpretability is not".
  - DIAG 2: low-data shows a real R-S accuracy gap (~3-4 pts) -> kept as corroboration.
  - DIAG 3: on raw STFT the CNN attends to the ~5 kHz resonance ridge, not fault lines;
    fault-band saliency ~1% == negative control -> switched to the envelope spectrum
    (0-500 Hz, 5.86 Hz resolution).
  - DIAG 4: Grad-CAM margin now measurable; ILG sign was OPPOSITE to the initial guess.
    Redefined ILG = margin(R) - margin(S) = +0.091 and documented the memorization
    interpretation instead of hiding the surprise.
- Verification performed: each diagnostic re-run from a saved script; conclusions match
  the committed diagnostic outputs.
- Errors found: the two design flaws above (accuracy saturation, wrong representation)
  were caught here and corrected rather than papered over.

## Entry 4 — Final experiments and statistics (analysis)
- Date: 2026-07 (analysis phase)
- Tool/model: Cursor agent (Claude)
- Objective: produce the headline results with proper statistics on two datasets.
- Files/sources supplied: envelope-cache + splits for DE and FE, seeds 0-9.
- Prompt summary: "Run R/S over ten seeds on DE and FE. Report mean +/- std, and add
  paired-t, Wilcoxon, and a seed-level bootstrap 95% CI on the ILG. Do not overstate."
- Output used: `main_results.json`, `summary.json`, `stats.json` (+ FE_*),
  `intra_record_similarity.json`, `within_vs_cross_margin.json`, `ablations.json`,
  figures including `fig_gradcam_qualitative.png`.
- Verification performed: paper numbers regenerated from these JSONs; ILG CI checked to
  exclude 0; per-seed values tabulated in the appendix for transparency.
- Errors found: none new; earlier 3-seed runs were superseded by the 10-seed runs and
  all documents updated for consistency.

## Entry 5 — Writing, review, and integrity
- Date: 2026-07 (writing/review phase)
- Tool/model: Cursor agent (Claude)
- Objective: draft the paper from evidence only, then simulate critical review and fix
  integrity issues.
- Files/sources supplied: all `results/tables/*.json`, `claim_evidence_table.md`, the
  ARS artifacts, `refs.bib`.
- Prompt summary: "Draft strictly from committed evidence. Then act as a hostile
  reviewer: find unsupported claims, citation errors, leakage/reproducibility risks, and
  propose fixes."
- Output used: `paper/main.tex` / `main.md`; revision response; integrity reports.
- Verification performed: verified citations against sources via web search; corrected
  the Lu et al. entry (title/authors/DOI) and labelled Vieira et al. as a non-peer-
  reviewed arXiv preprint; reconciled environment versions with the installed toolchain;
  aligned seed counts (10) across all documents.
- Errors found and fixed: misattributed M1-M7 (course material, not Lu et al.);
  inaccurate Lu et al. citation; missing preprint label; stale 3-seed mentions; missing
  dataset license/provenance block; leftover `__pycache__` / `spec_cache.npz`.

---

## Reusable prompt templates
The general template and the six universal prompt requirements are in
`Final_Experiment/PROMPT_PROTOCOL.md`. Module-specific prompts (research question,
leakage audit, method card, mechanism, results analysis, drafting, review, integrity)
are in `innovation_hints_cn.md` sections 7.1-7.8.

## Forbidden behavior (self-check)
No prompt in this project asked the model to fabricate results, write around failed
experiments, invent citations, hide AI use, inflate claims, or ignore leakage. The
DIAG entries above are the evidence: failures were reported and the design was changed.
