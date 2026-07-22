# CWRU Working Paper: Accuracy Is Blind to Leakage, Interpretability Is Not

A reproducible study contrasting a random-window split (R) and a recording-level split
(S) on CWRU bearing diagnosis, using an envelope-spectrum 1D-CNN. Main finding: on full
data, leakage inflates accuracy by only ~4.5 points but inflates the apparent physical
consistency of Grad-CAM explanations by an Interpretability Leakage Gap (ILG) of +0.095
(95% CI [0.079, 0.110], paired-t p=1.1e-6 over ten seeds),
driven by memorization of seen recordings.

## Research question
How does the train/test split protocol change apparent accuracy vs the apparent
physical consistency of Grad-CAM on CWRU? See `ars/research_question_brief.md`.

## Dataset
CWRU 12 kHz drive-end, 4 classes (Normal/IR/OR/B), 62 usable recordings. Raw .mat files
live in `../cv2/2025E/cwru_experiment/Source_Datasets` (paths set in `src/config.py`).
Metadata reused from the audited `file_metadata.json`. See `ars/leakage_audit.md`.

## Environment
```
python -m venv .venv   # Python 3.12
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r environment/requirements.txt
```

## Reproduce (run from repo root, with the workspace .venv active)
Primary dataset (DE, drive-end):
```
python src/make_manifest.py          # build data_manifest/manifest.csv
python src/preprocess.py             # cut windows -> windows.npz
python src/make_splits.py            # protocols R and S (verifies overlap R=62, S=0)
python src/mech_intra_record.py      # mechanism A: intra-record similarity
python src/run_experiments.py        # main: R vs S x 10 seeds -> summary.json + ILG
python src/stats_ilg.py --dataset all # paired-t + Wilcoxon + bootstrap CI -> stats.json
python src/evidence_within_cross.py  # memorization framing (seen vs unseen)
python src/run_ablations.py          # low-data, grouping, bandpass ablations (10 seeds)
python src/make_figures.py           # DE figures
```
Second dataset (FE, fan-end) for transfer, then cross-dataset figure:
```
python src/make_manifest.py --dataset FE
python src/preprocess.py   --dataset FE
python src/make_splits.py  --dataset FE       # verifies overlap R=48, S=0
python src/run_experiments.py --dataset FE    # FE_summary.json + ILG
python src/run_ablations.py   --dataset FE
python src/make_cross_figures.py              # fig_cross_dataset.png
```
Build the PDF (bibtex resolves the verified citations in `paper/refs.bib`):
```
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Key outputs
- `results/tables/summary.json`, `FE_summary.json` — accuracy, PC margin, ILG, gap (DE/FE)
- `results/tables/main_results.json`, `FE_main_results.json` — per (protocol, seed) metrics
- `results/tables/ablations.json`, `FE_ablations.json` — low-data / grouping / bandpass (10 seeds)
- `results/tables/stats.json`, `FE_stats.json` — paired-t, Wilcoxon, bootstrap 95% CI on ILG
- `results/tables/within_vs_cross_margin.json` — memorization evidence
- `results/figures/*.png` — similarity, R-vs-S bars, decision matrix, confusion, cross-dataset
- `paper/main.pdf`, `paper/main.tex`, `paper/main.md` — the working paper (22 pages, with appendices)
- `paper/refs.bib` — verified background citations (see `ars/literature_matrix.md`)
- `claim_evidence_table.md` — every claim mapped to an artifact

## Runtime assumptions
CPU only; the ten-seed pipeline runs in a few hours on a laptop. Seeds {0..9} fixed.

## Known limitations
CWRU is a lab benchmark with seeded faults; one dataset, one representation. Normal has
only 3 usable recordings. See `paper/main.md` Section 11 and the Threats to Validity
section.

## AI-use disclosure
See `paper/main.md` Section 14 and `ars/integrity_report.md` (M1-M7 self-audit).
