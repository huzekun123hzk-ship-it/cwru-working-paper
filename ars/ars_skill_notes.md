# ARS Skill Notes

Answers to manual 2.1, adapted to a tool-agnostic workflow (no Claude Code required).

1. What can ARS help with?
   Structuring the research-to-paper pipeline: research question, literature matrix,
   methodology blueprint, integrity gates, simulated review, revision, final audit.

2. What can ARS not do?
   It cannot invent valid results, choose the scientific claim, design the split, or
   verify citations for us. The human owns evidence and interpretation.

3. What are its integrity gates?
   Stage 2.5 (pre-writing integrity) and Stage 4.5 (final integrity), implemented here
   as `integrity_report.md` (M1-M7) and `final_integrity_report.md` (manual 16.1).

4. How does it reduce hallucinated citations / unsupported claims?
   By requiring every claim to map to an artifact (claim_evidence_table.md) and marking
   unverified references [CITATION NEEDED] and missing data [MATERIAL GAP].

5. What must the human still decide?
   Research question, dataset, split protocol, experimental evidence, verified
   citations, interpretation, and limitations.

This project applied the ARS logic manually per innovation_hints_cn.md 7.9.
