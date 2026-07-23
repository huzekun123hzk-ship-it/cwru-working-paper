# Revision Response

| Reviewer Concern | Response | Manuscript Change | Evidence |
|---|---|---|---|
| R1-1 Small accuracy gap could overstate | Agreed; abstract and Section 7 state the gap is only 5.3 pts and that the main effect is on interpretability. | Abstract, Sec 1, Sec 7, Sec 11 | summary.json |
| R1-2 ILG sign needs a direct control | Added seen-vs-unseen recording margin framing (R=100% seen, S=0% seen). | Sec 8, within_vs_cross_margin.json | within_vs_cross_margin.json |
| R1-3 Single-seed ablations | Resolved: all experiments (main, ablations B/C, mechanism) re-run with ten seeds; ILG significant (DE paired-t p=1.1e-6, FE p=0.008). | Sec 7, Sec 9, Sec 11, stats.json | ablations.json, FE_ablations.json, stats.json |
| R2-1 Nominal rpm band error | Added negative control (120 Hz shift) and stated residual risk in Limitations. | Sec 4, Sec 5, Sec 11 | main_results.json |
| R2-2 Representation validity | Reported envelope fault-band energy per class (IR 0.79 > Normal 0.47). | Sec 5 | diag_envelope_pc stdout |
| R3-1 Per-class recall / confusion | Added confusion matrices for both protocols. | Sec 7, fig_confusion | main_results.json, fig_confusion.png |
| R3-2 Classifier is a vehicle | Stated explicitly that the classifier is fixed and not a contribution. | Sec 4, Sec 11 | method_card.md |

Open items: all closed.
- Multi-seed: DONE (ten seeds across the main experiment and all ablations, both datasets).
- Citation verification: DONE. All seven references were verified against their primary
  sources and are in `paper/refs.bib`. During this check the Lu et al. (2026) entry was
  found to have an inaccurate title and incomplete author list; it was corrected to the
  true record ("Towards end-to-end automation of AI research", Lu, Lu, Lange, Yamada, Hu
  et al., Nature 651:914-919, doi:10.1038/s41586-026-10265-5), and the M1-M7 checklist was
  re-attributed to the course material rather than to that paper. Vieira et al. is labelled
  as a non-peer-reviewed arXiv preprint. No [CITATION NEEDED] markers remain.
