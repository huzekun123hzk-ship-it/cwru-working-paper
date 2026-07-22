# Research Question Brief

## Title
Accuracy is blind to leakage, interpretability is not: quantifying the interpretability leakage gap on CWRU bearing diagnosis

## Research question
Under an identical envelope-spectrum classifier, how does the train/test split protocol (random-window vs recording-level) change (a) apparent classification accuracy and (b) the apparent physical consistency of Grad-CAM explanations on the CWRU bearing dataset?

## Problem
CWRU signals are segmented into overlapping windows. Random-window splitting places near-identical adjacent windows from the same recording into both train and test, letting a model exploit recording identity rather than fault mechanism (manual 6.1).

## Bottleneck
Leakage-prone evaluation protocol.

## Hypothesis
Random-window leakage inflates not only accuracy but also the *apparent* interpretability of the model: because the model has memorized the specific recordings that reappear in the test set, its Grad-CAM saliency looks sharply concentrated on fault-frequency bands. Under a leakage-safe recording-level split, on unseen recordings, that apparent concentration drops. We name the difference the Interpretability Leakage Gap (ILG).

## Dataset(s)
CWRU 12 kHz drive-end (DE) vibration, 4 classes (Normal, IR, OR, B). 62 usable recordings after dropping 2 unreadable files. Bearing SKF6205.

## Method family
Fixed representation + classifier (envelope spectrum + 1D-CNN). The independent variable is the split protocol only. Grad-CAM physical consistency with a frequency-shift negative control.

## Baselines
- Protocol R (random-window) — the risk reference.
- Protocol S (recording-level) — the leakage-safe evaluation.
- Frequency-shifted band as a negative control for Grad-CAM.

## Mechanism validation
1. Intra-record window similarity: adjacent same-recording windows vs random cross-recording pairs (spectral cosine).
2. Envelope-spectrum fault-band energy fraction per class (physics of the representation).
3. Grad-CAM saliency fraction inside true fault bands vs shifted bands.

## Metrics
Accuracy, Macro-F1, per-class recall, confusion matrix; Grad-CAM margin (PC_true - PC_neg); ILG = margin(R) - margin(S); generalization gap = acc(R) - acc(S). Mean +/- std over 10 seeds, with paired-t/Wilcoxon tests and a seed-level bootstrap 95% CI on the ILG.

## Expected contribution
A named, reproducible metric (ILG) showing that leakage inflates interpretability appearance, plus the mechanistic explanation (memorized recordings) and a deployment decision matrix.

## Main risks
- Full-data accuracy may saturate (both protocols near 100%), hiding the accuracy gap. Mitigation: report the saturation honestly and add a low-data corroboration.
- Fault frequencies rely on nominal rpm; band definition error. Mitigation: shifted-band negative control.

## What would falsify the hypothesis
If margin(R) <= margin(S) within seed noise, or if the negative control is not separated from the true band, the hypothesis that leakage inflates interpretability appearance is not supported.
