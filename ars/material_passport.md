# Material Passport

## Data
- Source: CWRU Bearing Data Center, 12 kHz .mat files.
- Location: ../cv2/2025E/cwru_experiment/Source_Datasets (reused, previously downloaded).
- Metadata: references/file_metadata.json (161 files audited).
- Primary dataset DE (12k drive-end): 64 recordings, 62 usable (2 unreadable dropped),
  7533 windows.
- Second dataset FE (12k fan-end, different sensor location): 49 recordings, 5864 windows.
- Bearing: SKF6205 (references/bearing_params.json).

## Data provenance and license (per STUDENT_PRIVACY_AND_IP_RULES)
- Dataset URL: https://engineering.case.edu/bearingdatacenter (download page:
  https://engineering.case.edu/bearingdatacenter/download-data-file).
- Provider: Case Western Reserve University Bearing Data Center, Case School of Engineering.
- License / terms of use: publicly provided by CWRU for research use; the Bearing Data
  Center distributes the files freely and asks that the source be acknowledged. There is
  no explicit open-source license (e.g. no CC/MIT); use is governed by CWRU's general
  policies (case.edu legal notice). We do not redistribute the raw .mat files in this
  repository; we ship only a manifest and derived artifacts, and cite the source.
- Version: the Bearing Data Center files are unversioned; identity is pinned by SHA/path
  via references/file_metadata.json (161 files audited).
- Download date: not independently recorded; the raw files were reused from prior
  coursework (../cv2/2025E/cwru_experiment), downloaded from the CWRU site in that work.
  Flagged as a provenance gap: a fresh download with a recorded date would close it.
- Recommended dataset citation: Case Western Reserve University, Bearing Data Center.
  CWRU Bearing Fault Dataset [Dataset]. Case School of Engineering, CWRU.
  https://engineering.case.edu/bearingdatacenter
- Preprocessing and split design: see paper Sec 3-4, make_manifest.py / preprocess.py /
  make_splits.py; split protocols R and S with verified overlaps (R=62, S=0).

## Derived artifacts (committed)
- data_manifest/manifest.csv, windows.npz, splits_R.json, splits_S.json, env_cache.npz
- results/tables/*.json, results/figures/*.png, results/logs/run_log.txt

## Code
- src/*.py (config, io_cwru, preprocess, make_splits, features, envelope, models,
  engine, fault_freq, gradcam, mech_intra_record, run_experiments,
  evidence_within_cross, run_ablations, stats_ilg, make_figures, make_cross_figures,
  make_gradcam_fig)
- Diagnostic scripts retained for transparency: diag_difficulty.py, diag_hard.py,
  diag_saliency.py, diag_envelope_pc.py

## Environment
Python 3.12.8, torch 2.9.1+cpu, scikit-learn 1.8.0, numpy 2.4.0, scipy 1.16.3, matplotlib 3.10.8.

## Derived artifacts (updated)
- results/tables/*.json now includes stats.json, FE_stats.json (paired-t, Wilcoxon,
  bootstrap 95% CI on the ILG); results/figures now includes fig_gradcam_qualitative.png.

## Provenance note
Every number in paper/main.md (ten-seed results) is reproducible from the committed
scripts + data.
