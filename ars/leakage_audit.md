# Leakage Audit

Dataset: CWRU 12 kHz drive-end, 4 classes. Segmentation: 2048-point windows, hop 1024 (50% overlap), max 200 windows/recording. recording_id = one .mat file.

## Checks (manual 6.3)

| Risk | Path | Status | Mitigation |
|---|---|---|---|
| Adjacent-window leakage | Overlapping windows from one recording split across train/test | CONFIRMED as the object of study | Protocol S groups by recording_id; Protocol R kept only as risk reference. |
| Same recording in multiple splits | random assignment of windows | CONFIRMED for R (overlap=62 recordings), ELIMINATED for S (overlap=0) | `make_splits.py` group-by-recording; verified overlap train∩test=0 for S. |
| Normalization uses test data | mean/std over full set | AVOIDED | Stats computed on train indices only (`datautil.make_env_datasets`). |
| Augmentation leaks test info | frequency-axis roll/flip | AVOIDED | No frequency-axis augmentation; would break physical band localization (ch45). |
| Load/size hidden shortcut | class correlates with operating condition | PARTIALLY PRESENT | Ablation by load and fault_size (see ablations.json); fault_size split collapses to 0.47, reported as finding. |
| Cross-class file contamination | one file with multiple labels | NOT PRESENT | Each .mat has a single fault_type in metadata. |

## Verified split composition
- Protocol R: train∩test recording overlap = 62 (all recordings appear in train).
- Protocol S: train∩test recording overlap = 0.
Source: `make_splits.py` stdout, reproduced in run log.

## Dropped data
2 unreadable .mat files skipped (Normal_3HP, IR007_1HP); Normal has 3 usable recordings. Documented as a limitation (small Normal recording count).

Risk level overall: the study intentionally contrasts a HIGH-risk protocol (R) with a LOW-risk protocol (S). All other leakage paths are controlled.
