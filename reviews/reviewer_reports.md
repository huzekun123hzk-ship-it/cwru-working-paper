# Simulated Reviewer Reports

## Reviewer 1 (methodology)
Summary: The paper studies split protocol as the independent variable and proposes ILG.
The design is clean and the leakage control is verified in code.
Major concerns:
1. Full-data accuracy gap is small; the headline could be seen as weak. -> Ensure the
   abstract does not overstate the accuracy effect.
2. ILG sign is counterintuitive; the memorization explanation needs a direct control.
3. Grouping-by-fault_size and bandpass ablations are single-seed.
Minor: define PC_true/PC_neg earlier; clarify frequency-axis orientation.
Decision: minor-to-major revision.

## Reviewer 2 (signal processing / physics)
Summary: Envelope-spectrum choice is well justified by the STFT saliency diagnostic.
Major concerns:
1. Fault frequencies use nominal rpm; band error possible. -> Negative control helps,
   state residual risk.
2. Report envelope fault-band energy per class to establish representation validity.
Minor: bandpass 2-4 kHz selection rationale; Normal has few recordings.
Decision: minor revision.

## Reviewer 3 (ML evaluation)
Summary: Good use of seeds and train-only normalization. The decision matrix is a nice
summary.
Major concerns:
1. Provide per-class recall and confusion matrices for both protocols.
2. Clarify that the classifier is a fixed vehicle, not a contribution.
Decision: minor revision.
