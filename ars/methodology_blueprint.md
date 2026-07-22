# Methodology Blueprint

## Independent variable
Split protocol: R (random-window) vs S (recording-level). Everything else fixed.

## Pipeline
raw .mat (DE) -> windows (2048/1024) -> envelope spectrum (bandpass 2-4kHz -> Hilbert
-> FFT -> 0-500Hz) -> train-only z-score -> 1D-CNN -> metrics + Grad-CAM margin.

## Experiment matrix
All experiments use ten seeds (0-9) on both datasets (DE, FE). Statistical inference
(paired-t, Wilcoxon, seed-level bootstrap 95% CI) is in stats.json / FE_stats.json.
| # | Representation | Model | Protocol | Seeds | Primary metric | Non-accuracy metric | Artifact |
|---|---|---|---|---|---|---|---|
| E1 | envelope | 1D-CNN | R | 0-9 | accuracy, Macro-F1 | Grad-CAM margin | main_results.json |
| E2 | envelope | 1D-CNN | S | 0-9 | accuracy, Macro-F1 | Grad-CAM margin | main_results.json |
| M1 | raw window | - | - | - | spectral cosine | - | intra_record_similarity.json |
| A1 | envelope | 1D-CNN | R,S low-data | 0-9 | accuracy gap | - | ablations.json |
| A2 | envelope | 1D-CNN | group by rec/load/size | 0-9 | accuracy | - | ablations.json |
| A3 | envelope no-bandpass | 1D-CNN | S | 0-9 | accuracy | Grad-CAM margin | ablations.json |

## Derived quantities
- generalization_gap = acc(R) - acc(S)
- margin = PC_true - PC_neg
- ILG = margin(R) - margin(S)

## Controls
- Negative control: 120 Hz frequency-shifted fault bands.
- Memorization control: seen (R) vs unseen (S) recording margin.
