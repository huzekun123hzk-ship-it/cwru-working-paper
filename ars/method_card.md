# Method Card

Method name: Split-protocol contrast with envelope-spectrum Grad-CAM physical consistency (ILG)

Problem addressed: Leakage-prone random-window evaluation on CWRU inflates reported quality; accuracy alone cannot detect it.

Hypothesis: Random-window leakage inflates both accuracy and the apparent physical consistency of Grad-CAM; a recording-level split reveals the true (lower) interpretability on unseen recordings.

Input: 2048-point DE windows (hop 1024), converted to an envelope spectrum: bandpass 2-4 kHz resonance band -> Hilbert envelope -> FFT magnitude -> keep 0-500 Hz (86 bins, 5.86 Hz resolution).

Output: 4-class label (Normal, IR, OR, B) + Grad-CAM saliency over the envelope-spectrum frequency axis.

Key parameters: win=2048, hop=1024, resonance band=(2000,4000) Hz, env fmax=500 Hz, band halfwidth=15 Hz, negative-control shift=120 Hz, 1D-CNN (16-32-64) + GAP + linear, Adam lr=1e-3, 20 epochs, batch 64, class-weighted CE, seeds {0..9} (ten).

Why it should work: Bearing faults excite high-frequency resonance modulated at fault characteristic frequencies (BPFO/BPFI/BSF). The envelope spectrum recovers those low-frequency lines at high resolution, making "saliency inside fault band" a meaningful physical-consistency measure (unlike raw STFT, where the CNN keys on the 5 kHz resonance ridge).

Baselines: Protocol R (random-window, risk reference) and Protocol S (recording-level, safe). Frequency-shifted negative control for Grad-CAM.

Ablations: (A) low-data corroboration K in {10,25,50} windows/recording; (B) grouping variable recording vs load vs fault_size; (C) envelope without resonance bandpass.

Mechanism validation: intra-record window similarity; envelope fault-band energy per class; Grad-CAM true-band vs shifted-band fraction; within-record (seen) vs cross-record (unseen) margin.

Expected failure case: If both protocols saturate in accuracy AND Grad-CAM margins coincide, the ILG effect is absent (report as negative result).

Implementation risks: normalization leakage (mitigated: train-only stats), nominal-rpm band error (mitigated: negative control), bandpass band choice (mitigated: bandpass ablation).

Physical interpretation: A high-accuracy model on protocol R with inflated Grad-CAM focus is memorizing recordings, not learning fault physics; protocol S exposes the honest interpretability level on unseen bearings.

Evidence required before claiming success: ILG > 0 with 10-seed statistical significance (paired-t + bootstrap CI excluding 0), negative control separated from true band, and a mechanistic account (within>cross margin, intra-record similarity high). Achieved: DE ILG +0.095 (p=1.1e-6), FE +0.048 (p=0.008).
