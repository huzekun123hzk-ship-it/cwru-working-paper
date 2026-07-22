# Final Integrity Report

Audit against manual 16.1 questions.

| # | Question | Result | Note |
|---|---|---|---|
| 1 | Every result exists in logs/tables/figures? | PASS | All numbers trace to results/tables/*.json and run_log.txt. |
| 2 | Every citation exists and supports the sentence? | PASS | 7 background citations verified against primary sources in `paper/refs.bib` (Smith&Randall 2015, Hendriks 2022, Vieira 2025, Randall&Antoni 2011, Selvaraju 2017, Adebayo 2018, Lu 2026); no result depends on them. |
| 3 | Dataset claims traceable to source? | PASS | Sampling rate, classes, counts from audited file_metadata.json. |
| 4 | AI-generated claims verified? | PASS | Numbers re-read from artifacts; ILG recomputed via --agg-only. |
| 5 | Limitations explicit? | PASS | Sec 11 lists 6 limitations incl. CWRU lab-benchmark caveat. |
| 6 | Split design clearly described? | PASS | Sec 3; overlaps verified R=62, S=0. |
| 7 | Negative/failed results honestly reported? | PASS | Accuracy saturation, opposite-sign ILG surprise, STFT failure all reported. |
| 8 | AI-use disclosure complete? | PASS | Sec 14 + prompt_log.md + this audit. |

## Blocking issues
None for the working-paper stage.

## Non-blocking issues
- Background citations are now verified in `paper/refs.bib`; extend the literature review
  further only if targeting external submission.
- Transfer is shown across CWRU sensors (DE, FE) only; a physically distinct machine
  (e.g. gearbox) would further strengthen generality.

## Overall
PASS (working-paper stage). The paper's conclusions rest only on verified,
reproducible artifacts. All experiments are 10-seed on both datasets; the ILG is
statistically significant (DE paired-t p=1.1e-6, 95% CI [0.079,0.110]; FE p=0.008,
95% CI [0.022,0.073]), and the second-dataset (FE) transfer is confirmed with positive
accuracy gap and ILG.
