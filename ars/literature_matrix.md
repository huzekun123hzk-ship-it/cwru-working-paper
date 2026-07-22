# Literature Matrix

All background citations have been verified against their primary sources and are
collected in `paper/refs.bib` (rendered via BibTeX, `ieeetr` style). We do not invent DOIs.

| Topic | Source | What it supports | Status |
|---|---|---|---|
| CWRU benchmark | Smith & Randall, MSSP 64-65:100-131, 2015 (doi:10.1016/j.ymssp.2015.04.021) | CWRU as standard benchmark; records vary in difficulty | VERIFIED |
| CWRU leakage (canonical) | Hendriks, Dumond & Knox, MSSP 169:108732, 2022 (doi:10.1016/j.ymssp.2021.108732) | condition-wise split reuses same physical bearings -> memorization; proposes bearing-independent benchmarking | VERIFIED |
| Leakage across datasets | Vieira, Bauler, Rosa & Silva, arXiv:2509.22267, 2025 (doi:10.48550/arXiv.2509.22267) | segment/condition-wise splits inflate accuracy by tens of pts on CWRU/PU/Ottawa; recommend bearing/recording-level splits | VERIFIED (non-peer-reviewed preprint) |
| Envelope analysis / HFRT | Randall & Antoni, MSSP 25(2):485-520, 2011 (doi:10.1016/j.ymssp.2010.07.017) | envelope analysis as benchmark demodulation method | VERIFIED |
| Fault characteristic freqs | bearing kinematics (SKF6205) | BPFO/BPFI/BSF formulas | reused from cv2 bearing_params.json |
| Grad-CAM | Selvaraju et al., ICCV 2017, pp. 618-626 (doi:10.1109/ICCV.2017.74) | class-discriminative saliency | VERIFIED |
| M1-M7 integrity checklist | course material (innovation_hints_cn.md 7.10 / ARS integrity gate) | the seven failure modes we self-audit against | provided by course material (not an external paper) |
| Automated AI-research risks | Lu, Lu, Lange, Yamada, Hu et al., "Towards end-to-end automation of AI research", Nature 651:914-919, 2026 (doi:10.1038/s41586-026-10265-5) | context: automated research produces hallucinated citations etc. | VERIFIED (title/authors/DOI checked against Nature) |

Verification note: the results and methods in this paper do not depend on any
citation; citations support background framing only. Each reference above was located
via its DOI / venue and its bibliographic fields cross-checked before inclusion.
Vieira et al. is an arXiv preprint (not peer-reviewed) and is labelled as such in the
manuscript. Correction log: the Lu et al. entry was previously recorded with an
inaccurate title ("Analysis of failure modes...") and an incomplete author field; on
re-verification against Nature the true title is "Towards end-to-end automation of AI
research" (Lu, Lu, Lange, Yamada, Hu et al.), and the M1-M7 checklist is attributed to
the course material rather than to that paper.
