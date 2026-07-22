# Integrity Report (M1-M7 self-audit)

Reference: the seven failure modes (M1-M7) are defined in the course material
(innovation_hints_cn.md 7.10 / ARS integrity gate), not in an external paper.
Status codes: CLEARED (checked, no issue) / SUSPECTED (open) / N/A.

| ID | Mode | Check in this project | Status |
|---|---|---|---|
| M1 | Implementation bug passing self-review | Normalization uses train-only stats; split overlap verified (S=0, R=62); metrics via scikit-learn. Grad-CAM verified against ch53 concept and negative control. | CLEARED |
| M2 | Hallucinated citation | 7 background citations verified against primary sources and collected in `paper/refs.bib` (rendered via BibTeX); each was located by DOI/venue before inclusion. No result depends on any citation. | CLEARED |
| M3 | Hallucinated results | Every number in the paper is traceable to results/tables/*.json or run logs. No placeholder numbers. | CLEARED |
| M4 | Shortcut reliance | This is the object of study. Demonstrated via intra-record similarity + within/cross margin + fault_size collapse. | CLEARED (studied) |
| M5 | Bug framed as discovery | The ILG sign was OPPOSITE to the initial hypothesis; we did not hide it. We re-derived the honest interpretation (memorization inflates apparent focus) and redefined ILG=margin(R)-margin(S) transparently. Full-data accuracy saturation reported honestly. | CLEARED |
| M6 | Methodology fabrication | Recording-level split is actually implemented and its overlap=0 is verified in code output, not merely claimed. | CLEARED |
| M7 | Early framing lock-in | Initial "leakage inflates accuracy" framing was tested, found weak on full data, and revised after diagnostics (difficulty + saliency-distribution). Multiple representations (STFT then envelope) were tried before committing. | CLEARED |

## Key honesty notes
1. Full-data accuracy does NOT show large leakage inflation; reported as such (claim #3).
2. Original hypothesis direction was falsified for PC_true alone; the supported claim uses the Grad-CAM margin and is stated with bounded wording.
3. All experiments (main, ablations, mechanism) are now 10-seed on BOTH datasets. The bandpass ablation shows the margin collapses without the resonance bandpass (DE 0.010, FE -0.033). The ILG is statistically significant: DE +0.095 (95% CI [0.079,0.110], paired-t p=1.1e-6, 10/10 seeds positive), FE +0.048 (95% CI [0.022,0.073], p=0.008, 9/10). See stats.json.
4. Second dataset (FE) added; both the accuracy gap and ILG remain positive, so the transfer claim is bounded to CWRU sensors (not a distinct machine).

Overall: PASS. All ablations multi-seed; transfer verified on a second sensor.
