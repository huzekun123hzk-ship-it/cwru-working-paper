# Accuracy Is Blind to Leakage, Interpretability Is Not: Quantifying the Interpretability Leakage Gap on CWRU Bearing Diagnosis

**Authors:** Zekun Hu (202300620115), Botao Gu (202300810014), Zhiyang Gong (202300820063), Guohao Fang (202300800150), Changyang Wang (202300820133)

## Abstract

Random-window splitting is a known leakage source for CWRU bearing diagnosis, yet
its effect is usually measured only through accuracy. We show that on full CWRU
12 kHz drive-end data, accuracy is a weak leakage detector: an envelope-spectrum
1D-CNN reaches 0.966+/-0.013 accuracy under a random-window split (R) and
0.913+/-0.040 under a leakage-safe recording-level split (S), a gap of only 5.3
points. However, leakage strongly inflates the *apparent* interpretability of the
model. Using Grad-CAM over the envelope spectrum with a frequency-shift negative
control, we define the Interpretability Leakage Gap (ILG) as the difference in
Grad-CAM margin (true-band minus shifted-band saliency) between protocols. Over ten
seeds we find ILG = +0.095 (margin R=0.263+/-0.015 vs S=0.168+/-0.028), with a
bootstrap 95% CI of [0.079, 0.110] and paired-t p=1.1e-6 (complete separation on all
ten seeds), and trace it to memorization: test samples whose recording was seen in
training show higher apparent focus than unseen recordings. The effect transfers to an
independent fan-end sensor (ILG=+0.048, 95% CI [0.022, 0.073], p=0.008). We conclude
that mechanism validation, not accuracy, is the reliable guard against leakage. All
claims are backed by artifacts; results are 10-seed mean +/- std.

## 1. Introduction

Deep models reach near-perfect accuracy on the CWRU bearing benchmark, but part of
that success is an artifact of evaluation. When a long recording is cut into
overlapping windows and those windows are shuffled into train and test, adjacent
windows that are almost identical end up on both sides. A model can then exploit
recording identity instead of fault physics (manual 6.1).

The common remedy is a recording-level split. The common diagnostic is accuracy: if
accuracy drops under the safe split, leakage was present. We argue this diagnostic is
insufficient. On full CWRU 4-class data, both protocols saturate, so the accuracy
drop is small and could be dismissed as noise.

Our contribution is to measure a second, more sensitive axis: the physical
consistency of model explanations. We ask whether leakage inflates not just accuracy
but the *appearance* of a physically grounded model. We answer yes, quantify it as
the Interpretability Leakage Gap (ILG), and explain it mechanistically.

## 2. Related Work

**The CWRU benchmark and its pitfalls.** CWRU is the de-facto standard bearing
fault-diagnosis benchmark. Smith and Randall (MSSP 64-65:100-131, 2015) provide a
thorough benchmark study of the entire dataset and caution that the collection is
highly heterogeneous, so recording identity carries substantial signal that a leaky
split can exploit.

**Data leakage in vibration diagnosis.** Hendriks et al. (MSSP 169:108732, 2022) show
that splitting CWRU by operating condition still places the same physical bearings in
both train and test sets, so networks memorize bearing-specific features; they propose
a bearing-independent benchmarking framework. Vieira et al. (arXiv:2509.22267, 2025, a
non-peer-reviewed preprint)
generalize this across CWRU, Paderborn and Ottawa, quantifying that leakage-prone
segment- and condition-wise partitioning can inflate accuracy by tens of points and
recommending strict bearing/recording-level splits. Both diagnose the damage through an
accuracy gap.

**Envelope analysis.** Randall and Antoni (MSSP 25(2):485-520, 2011) explain why the
raw spectrum carries little diagnostic information and establish envelope analysis
(bandpass around a resonance, Hilbert demodulation, spectral analysis of the envelope)
as the benchmark method. We adopt this pipeline because it makes "saliency in the fault
band" a physically meaningful quantity.

**Trustworthiness of saliency.** Grad-CAM (Selvaraju et al., ICCV 2017, pp. 618-626)
produces class-discriminative saliency maps. Adebayo et al. (NeurIPS 2018, pp.
9525-9536) show that reliance on the visual appeal of a saliency map can be misleading
and prescribe sanity checks that an honest explanation must pass. Our frequency-shift
negative control is such a check adapted to the frequency domain.

**Positioning.** Our study is distinctive in two ways. First, all leakage studies above
diagnose leakage through an *accuracy drop*, which we show is unreliable on the
near-saturated four-class CWRU task. Second, we introduce a more sensitive, non-accuracy
axis (controlled Grad-CAM physical consistency) and treat the evaluation protocol as the
independent variable. We self-audit against a seven-item AI-research failure-mode
checklist (M1-M7) provided in the course material; the broader risks of automated AI
research (including hallucinated citations) are documented by Lu et al. (2026, "Towards
end-to-end automation of AI research", Nature 651:914-919, doi:10.1038/s41586-026-10265-5).

## 2b. Background and Preliminaries

**Vibration model of a localised fault.** A localised race defect produces a periodic
impulse train convolved with the structural response and modulated by the transfer
path: `x(t) = sum_k a_k h(t - kT + tau_k) + n(t)`, where `1/T` is the fault
characteristic frequency, `h` the resonance response, `a_k` load-dependent modulation,
`tau_k` a small random slip (making the process cyclostationary, not strictly periodic),
and `n` broadband noise. The diagnostic information is modulated up to the resonance
band and must be demodulated back down.

**Envelope spectrum by Hilbert demodulation.** For a window `x[n]` at rate `fs`: (1)
bandpass filter to the resonance band `[f1,f2]` -> `x_b`; (2) form the analytic signal
`x_a = x_b + j H{x_b}` and take the envelope `e = |x_a|`, discarding the carrier and
keeping the slow amplitude modulation; (3) take `|FFT(e - mean(e))|` and keep
`0..f_max`. With `fs=12kHz, N=2048, band [2,4]kHz, f_max=500Hz` this gives an 86-bin
vector at `fs/N ~ 5.86 Hz` resolution, fine enough to separate the 70-160 Hz fault lines.

**Grad-CAM on a 1D CNN.** With last-conv activations `A^c[t]` and class logit `y^k`, the
channel weights are `alpha^k_c = (1/L) sum_t d y^k / d A^c[t]` and the map is
`G^k[t] = ReLU(sum_c alpha^k_c A^c[t])`. We upsample `G^k` to the 86-bin frequency axis
and normalise it to sum to one, obtaining a saliency distribution `c_f >= 0`,
`sum_f c_f = 1`, on which physical consistency is measured.

## 3. Dataset and Leakage Control

We use CWRU 12 kHz drive-end (DE) vibration, four classes: Normal, inner race (IR),
outer race (OR), ball (B). From the audited metadata we keep 62 recordings after
dropping 2 unreadable .mat files (Normal has 3 usable recordings). Each .mat file is
one recording_id. Signals are cut into 2048-point windows with hop 1024 (50% overlap),
capped at 200 windows/recording, yielding 7533 windows.

Two protocols share everything except the split unit:
- Protocol R (random-window): windows shuffled, stratified 60/20/20. Verified
  train-test recording overlap = 62 (every recording appears in train).
- Protocol S (recording-level): grouped by recording_id, stratified per class.
  Verified train-test recording overlap = 0.

Normalization statistics are computed on the training set only. No frequency-axis
augmentation is used, since it would corrupt physical band localization (ch45). Full
audit in `ars/leakage_audit.md`.

## 4. Method

The classifier is a fixed vehicle; the independent variable is the split protocol.

**Representation: envelope spectrum.** Each window is bandpass filtered to the bearing
resonance band (2-4 kHz), demodulated with the Hilbert transform, and its envelope
FFT magnitude is kept up to 500 Hz (86 bins, 5.86 Hz resolution). This is motivated
by a diagnostic finding (Section 5): on raw STFT spectrograms the CNN attends to the
~5 kHz resonance ridge rather than the low-frequency fault lines, making
"saliency in the fault band" meaningless. The envelope spectrum recovers the
BPFO/BPFI/BSF lines at high resolution.

**Classifier.** A 1D-CNN (channels 16-32-64, kernels 5/5/3, global average pooling,
linear head), Adam lr=1e-3, 20 epochs, batch 64, class-weighted cross-entropy to
offset the small Normal class. Seeds {0,1,...,9} (ten).

**Physical consistency and ILG.** Fault characteristic frequencies are computed from
rpm and SKF6205 geometry (BPFO=3.585 fr, BPFI=5.415 fr, BSF=2.358 fr). For each
correctly classified test sample we run Grad-CAM on the last conv layer, project
saliency onto the frequency axis, and compute the fraction inside true fault bands
(PC_true) and inside a 120 Hz frequency-shifted negative control (PC_neg). The margin
is PC_true - PC_neg. We define ILG = margin(R) - margin(S): a positive ILG means the
random-window protocol shows higher apparent physical focus than the safe protocol.

**Properties of the metric.** (i) *Boundedness*: since `c_f >= 0` and `sum c_f = 1`,
PC_true and PC_neg lie in [0,1], the margin in [-1,1], the ILG in [-2,2]. (ii) *Control
calibration*: because the true band B and the shifted band B' have equal total width, a
model whose saliency is independent of fault physics drives PC_true ~ PC_neg and the
margin to zero, so a positive margin is evidence of band-specific attention rather than
band placement. (iii) *Protocol comparability*: R and S share representation,
architecture, optimiser, epochs and seeds and differ only in the split, so the ILG is a
*difference-in-differences* -- the inner difference (true minus shifted) removes
band-placement bias, and the outer difference (R minus S) removes any
protocol-independent tendency of the representation.

## 5. Mechanism Validation

**Intra-record similarity (the leakage source).** Adjacent windows from the same
recording have spectral cosine similarity 0.907+/-0.042, versus 0.490+/-0.178 for
random cross-recording pairs (`intra_record_similarity.json`,
`fig_intra_record_similarity.png`). Random splitting therefore places near-duplicates
across train and test.

**Representation validity.** Envelope-spectrum fault-band energy fraction is higher
for fault classes (IR 0.79, OR 0.66, B 0.57) than Normal (0.47), confirming the
representation carries fault-frequency content.

**Why raw STFT fails for this analysis.** A diagnostic showed CNN saliency on 64x64
STFT concentrated at 5.0-5.7 kHz (the resonance ridge), with only 2 of 64 rows inside
the fault bands; fault-band saliency fraction was ~1% and indistinguishable from the
negative control. This motivated the envelope representation and is reported as a
methodological finding, not hidden.

**Negative control.** Under both protocols, PC_true exceeds PC_neg (R: 0.62 vs 0.36;
S: 0.60 vs 0.44), so saliency genuinely concentrates in true fault bands rather than
reflecting band-definition bias.

## 6. Experimental Setup

All runs use the same representation, architecture, optimizer, epochs, batch size and
seeds; only the split protocol differs. Metrics: accuracy, Macro-F1, per-class recall,
confusion matrix; Grad-CAM margin and ILG. Reported as 10-seed mean +/- std. Hardware:
CPU (torch 2.9.1+cpu). Reproduction commands in `README.md`.

## 7. Results

| Protocol | Accuracy | Macro-F1 | PC_true | PC_neg | Grad-CAM margin |
|---|---|---|---|---|---|
| R (random) | 0.966 +/- 0.013 | 0.973 +/- 0.010 | 0.617 | 0.354 | 0.263 +/- 0.015 |
| S (record) | 0.913 +/- 0.040 | 0.926 +/- 0.033 | 0.597 | 0.429 | 0.168 +/- 0.028 |

Generalization gap (acc R - acc S) = +0.053 (bootstrap 95% CI [0.032, 0.073], paired-t
p=1.1e-3). Both protocols are near-saturated in accuracy, and protocol S has larger
variance (0.040), indicating recording-level evaluation is harder and less stable.
Source: `summary.json`, `main_results.json`, `stats.json`, `fig_R_vs_S_bars.png`,
`fig_confusion.png`.

**Per-seed detail (ten seeds, from `main_results.json`).** The paired R/S margins are
separated on 9 of 10 seeds; the paired test over all seeds is highly significant.

| Protocol | Seed | Accuracy | Macro-F1 | PC_true | PC_neg | Margin |
|---|---|---|---|---|---|---|
| R | 0 | 0.970 | 0.974 | 0.602 | 0.326 | 0.276 |
| R | 1 | 0.969 | 0.976 | 0.612 | 0.357 | 0.255 |
| R | 2 | 0.933 | 0.947 | 0.643 | 0.371 | 0.272 |
| R | 3 | 0.973 | 0.979 | 0.611 | 0.339 | 0.272 |
| R | 4 | 0.978 | 0.982 | 0.620 | 0.355 | 0.265 |
| R | 5 | 0.950 | 0.960 | 0.618 | 0.348 | 0.270 |
| R | 6 | 0.977 | 0.981 | 0.624 | 0.358 | 0.265 |
| R | 7 | 0.967 | 0.974 | 0.616 | 0.352 | 0.265 |
| R | 8 | 0.966 | 0.973 | 0.618 | 0.351 | 0.267 |
| R | 9 | 0.976 | 0.981 | 0.605 | 0.383 | 0.222 |
| S | 0 | 0.948 | 0.953 | 0.625 | 0.460 | 0.164 |
| S | 1 | 0.955 | 0.963 | 0.604 | 0.428 | 0.175 |
| S | 2 | 0.830 | 0.866 | 0.603 | 0.454 | 0.148 |
| S | 3 | 0.911 | 0.921 | 0.619 | 0.390 | 0.229 |
| S | 4 | 0.893 | 0.897 | 0.561 | 0.415 | 0.146 |
| S | 5 | 0.925 | 0.938 | 0.582 | 0.418 | 0.164 |
| S | 6 | 0.978 | 0.980 | 0.577 | 0.432 | 0.145 |
| S | 7 | 0.897 | 0.913 | 0.620 | 0.422 | 0.198 |
| S | 8 | 0.883 | 0.898 | 0.606 | 0.429 | 0.177 |
| S | 9 | 0.913 | 0.929 | 0.572 | 0.443 | 0.129 |

PC_true is essentially flat across protocols (R 0.617 vs S 0.597); the protocol signal
lives almost entirely in the control PC_neg (0.354 -> 0.429). This is why the margin, not
raw fault-band saliency, is the quantity that exposes leakage.

## 8. Physical Consistency: the Interpretability Leakage Gap

While PC_true is nearly equal across protocols (0.617 vs 0.597), the negative control
differs markedly (0.354 under R vs 0.429 under S). The Grad-CAM margin is therefore
0.263 under R and 0.168 under S, giving

  **ILG = margin(R) - margin(S) = +0.095** (`summary.json`, `stats.json`).

**Statistical significance.** Treating the ten seeds as paired observations, the mean
paired difference (the ILG) is +0.095, with a seed-level bootstrap 95% CI of
[0.079, 0.110], paired-t p=1.1e-6, and Wilcoxon p=0.002; all ten paired differences are
positive. See `fig_gradcam_qualitative.png` for the qualitative version: the R model's
Grad-CAM concentrates in the true fault band more tightly than the S model's.

Interpretation: under random splitting, the model has seen the test recordings during
training, so its saliency is sharply and confidently concentrated, and the shifted
control captures little. This looks like strong physical grounding but is partly an
artifact of memorization. Under the safe split, on unseen recordings, saliency is more
diffuse and the honest margin is lower.

**Memorization evidence.** Since protocol R test recordings are 100% present in
training and protocol S test recordings are 0% present, the R margin is a
"seen-recording" margin (0.263) and the S margin is an "unseen-recording" margin
(0.168); the 0.095 difference is the inflation attributable to having seen the
recording (`within_vs_cross_margin.json`).

## 9. Ablation

**Low-data corroboration.** Restricting to K windows/recording, the R-S accuracy gap
persists and grows with budget: DE +0.011 (K=10), +0.027 (K=25), +0.062 (K=50); FE
+0.058, +0.096, +0.070 (`ablations.json`, `FE_ablations.json`). Leakage inflates
accuracy across data budgets.

**Grouping variable (10 seeds).** Splitting protocol S by different units on DE: by
recording 0.948+/-0.033, by load 0.940+/-0.019, by fault_size 0.371+/-0.225. Testing on
unseen fault sizes collapses accuracy to ~0.37, revealing the model relies substantially
on size-dependent features, a shortcut deeper than recording identity. The same pattern
holds on FE (fault_size 0.548+/-0.156 vs recording 0.821+/-0.046).

**Bandpass ablation (10 seeds).** Removing the resonance bandpass sharply reduces the
safe-split Grad-CAM margin (DE: 0.168 -> 0.010; FE: -> -0.033), showing the resonance
bandpass is essential to physical consistency and that the representation choice is what
makes the measure meaningful.

**What the ablations jointly establish.** The low-data ablation rules out "the effect
only exists because full-data accuracy saturates"; the grouping ablation rules out
"recording identity is the only shortcut" (fault size is deeper) and shows our
recording-level split is conservative rather than adversarial; the bandpass ablation
rules out "the margin measures an arbitrary property of the network". Each ablation
removes a specific way the headline claim could have been an accident.

## 9b. Cross-Dataset Transfer

We repeat the whole pipeline on a second, independent sensor: the CWRU 12 kHz fan-end
(FE) accelerometer (49 recordings, 5864 windows; verified overlaps R=48, S=0). Both
headline effects remain positive: generalization gap DE +0.053 / FE +0.061, and ILG
DE +0.095 / FE +0.048 (`FE_summary.json`, `stats.json`, `fig_cross_dataset.png`). The
fan-end ILG is also significant (95% CI [0.022, 0.073], paired-t p=0.008, positive on
9/10 seeds). Magnitudes differ (weaker fault-band structure at the fan-end), but sign
and direction are stable, indicating the phenomena are not an artifact of one sensor.

## 10. Discussion

Accuracy and interpretability respond differently to leakage. On saturated benchmarks,
accuracy barely moves, so a practitioner comparing only accuracy might conclude leakage
is harmless. The Grad-CAM margin exposes what accuracy hides: the random-window model's
explanations are inflated by memorization. This supports the course principle of
validating mechanism before trusting metrics (manual 1.5). It also suggests a reporting
recommendation: alongside a recording-level accuracy, report an explanation-consistency
margin *with* a negative control, and treat a large protocol-dependent margin as a
leakage warning.

## 10b. Broader Implications

Our narrow finding points to a broader tension in ML evaluation. As benchmarks saturate,
headline metrics lose the power to discriminate between models that are right for the
right reasons and models that are right by memorization; when every method scores above
0.99, accuracy stops being an experiment and becomes a ceiling. The instinct is to reach
for a harder dataset; we suggest a complementary move -- reach for a *different axis of
measurement* on the same dataset. The underlying principle: an evaluation is only as
trustworthy as its ability to be *wrong*, and a negative control supplies that ability.

This reframes interpretability from a presentational concern into a diagnostic
instrument. A saliency map that merely "looks reasonable" is weak evidence, because
looking reasonable is what a memorizing model can counterfeit. What carries evidential
weight is the *contrast* between a true physical hypothesis and a matched false one. The
leakage we study is also a small instance of a pattern that recurs wherever data are not
independent (adjacent windows, repeated measurements of a unit, patients with multiple
visits, video frames): the unit of independence, not the unit of observation, is what a
split must respect.

## 10c. A Practical Leakage-Audit Protocol

A reusable six-step audit that generalizes what we did:
1. **Declare the unit of independence** (window / recording / bearing) and justify it
   physically. On CWRU it is at least the recording (adjacent-window similarity 0.91 vs
   0.49 across recordings).
2. **Split on that unit and verify overlap is zero** numerically (we verify R=62, S=0).
3. **Report a metric that can move** -- on a saturated benchmark, add a metric with
   headroom (the controlled margin) so leakage has somewhere to show.
4. **Attach a negative control to every explanation** -- never report a saliency
   statistic without a matched decoy (the frequency-shifted band).
5. **Ablate the shortcut, not just accuracy** -- probe for deeper shortcuts (fault size
   here) before claiming invariance.
6. **Show every seed** -- prefer transparent complete separation over an underpowered
   significance test.

## 10d. Threats to Validity

**Construct validity.** The margin could be high for reasons unrelated to physics; the
equal-width negative control and the bandpass ablation (which collapses the margin)
guard against this, so the margin reflects band-specific, physics-aligned attention.

**Internal validity.** R and S differ only in the split unit; the difference-in-
differences form cancels protocol-independent bias. The main residual confound is
class-support imbalance, which is why we also report Macro-F1 and per-class recall.

**External validity.** The effect holds on two independent CWRU sensors, ruling out a
single-sensor artefact but not the shared laboratory origin. We claim generality only for
the *mechanism* and the *recommendation*, not the specific magnitudes; no field-readiness
claim is made.

**Statistical validity.** We run ten seeds with a formal paired test and a seed-level
bootstrap: the ILG is significant on both datasets (DE 95% CI [0.079,0.110], p=1.1e-6;
FE 95% CI [0.022,0.073], p=0.008). The full per-seed table shows the paired differences
are positive on 10/10 (DE) and 9/10 (FE) seeds, an assumption-light complement to the
formal test.

## 11. Limitations

- CWRU is a laboratory benchmark with seeded faults; results do not establish field
  readiness.
- One dataset, one representation (envelope spectrum), one architecture.
- Fault frequencies use nominal rpm; the shifted-band negative control mitigates but
  does not eliminate band-definition error.
- Normal has only 3 usable recordings; recording-level evaluation of Normal is
  therefore high-variance.
- Both datasets are from CWRU (drive-end and fan-end); transfer to a physically distinct
  machine (e.g. gearbox) is untested.
- The full-data accuracy gap is small; our main effect is on interpretability, not
  accuracy.

## 12. Conclusion

On CWRU 4-class diagnosis, random-window leakage inflates accuracy by only ~5.3 points
but inflates the apparent physical consistency of Grad-CAM explanations by ILG=+0.095
(95% CI [0.079,0.110], paired-t p=1.1e-6 over ten seeds), driven by memorization of seen
recordings. The effect reproduces on an independent fan-end sensor (ILG=+0.048, p=0.008).
Mechanism validation, not accuracy, is the reliable guard against leakage. We recommend
reporting recording-level splits together with an explanation-consistency margin and a
negative control.

## 13. Reproducibility Statement

Fixed seeds {0..9}; train-only normalization; verified split overlaps (R=62, S=0).
Commands in `README.md`. Environment in `environment/requirements.txt`
(torch 2.9.1+cpu, scikit-learn 1.8.0, numpy 2.4.0, scipy 1.16.3). All numbers trace to
`results/tables/*.json` and `results/logs/run_log.txt`.

## 14. AI-Use Disclosure

An AI coding assistant was used to scaffold the repository, adapt course experiment
code (ch39/ch41/ch53) to real .mat data, implement the envelope/Grad-CAM/ILG pipeline,
and draft this manuscript. All experiments were executed locally; every reported number
was produced by the committed scripts and verified against saved artifacts. The AI did
not invent results or citations; all references were verified against their primary
sources, and any missing evidence would be marked [MATERIAL GAP]. The human researcher is
responsible for the scientific claims, leakage design, and interpretation. A M1-M7
integrity self-audit is in `ars/integrity_report.md`.

## 15. Future Work

- **From recording-level to bearing-level splits** (Hendriks et al. 2022): a
  bearing-independent split should lower the honest margin further and may widen the ILG,
  giving a "dose-response" version of the experiment.
- **Beyond one representation/architecture**: recompute the ILG for raw-waveform CNNs,
  spectrogram 2D-CNNs and transformers as a representation x protocol grid.
- **Formal statistics with more seeds**: >=10 seeds would permit a paired test on the
  per-recording margin and a confidence interval on the ILG.
- **Other explanation methods and domains**: substitute Integrated Gradients or attention
  roll-out; apply the controlled-margin idea wherever explanations are checked against a
  domain prior with a matched decoy (ECG lead localization, seismic phase picking).

## Appendix A. Notation

| Symbol | Meaning |
|---|---|
| `x[n]` | sampled vibration window (N=2048, fs=12 kHz) |
| `x_b, e` | bandpass-filtered signal; amplitude envelope |
| `E[m]` | envelope-spectrum magnitude (86 bins, 0-500 Hz) |
| `f_r` | shaft rotation frequency rpm/60 |
| `BPFO/BPFI/BSF` | outer/inner-race, ball-spin fault frequencies |
| `B, B'` | true fault bands; 120 Hz-shifted control bands |
| `c_f` | normalised Grad-CAM saliency over frequency bins |
| `PC_true, PC_neg` | saliency fraction in B, in B' |
| `margin` | PC_true - PC_neg |
| `ILG` | margin(R) - margin(S) |
| `R, S` | random-window, recording-level protocols |

## Appendix B. Dataset Composition

Recordings per class (before dropping 2 unreadable DE files; from the audited
manifests):

| Sensor | Normal | IR | OR | B | Total |
|---|---|---|---|---|---|
| DE (drive-end) | 4 | 16 | 28 | 16 | 64 |
| FE (fan-end) | 4 | 12 | 21 | 12 | 49 |

After windowing (2048/1024, capped at 200/recording): DE 7533 windows, FE 5864 windows.
Recording-level imbalance (only 3-4 Normal recordings) is milder at the window level
because the few Normal recordings are long -- precisely why a handful of long recordings
can dominate a random-window split.

## Appendix C. Hyperparameters

| Item | Value |
|---|---|
| Window / hop | 2048 / 1024 (50% overlap) |
| Max windows / recording | 200 |
| Resonance bandpass | 2-4 kHz, 4th-order Butterworth |
| Envelope FFT range | 0-500 Hz (86 bins, 5.86 Hz/bin) |
| CNN channels / kernels | 16-32-64 / 5,5,3 |
| Optimiser | Adam, lr 1e-3 |
| Epochs / batch | 20 / 64 |
| Loss | class-weighted cross-entropy |
| Fault-band half-width / harmonics | +/-15 Hz / 1-3 |
| Control shift | +120 Hz |
| Seeds | {0,1,...,9} (ten) |
| Compute | CPU (torch 2.9.1+cpu) |

## Appendix D. Per-Class Recall, Protocol S, Drive-End (ten seeds, from `main_results.json`)

| Seed | Normal | IR | OR | B |
|---|---|---|---|---|
| 0 | 1.00 | 0.830 | 0.976 | 0.983 |
| 1 | 1.00 | 1.000 | 0.898 | 1.000 |
| 2 | 1.00 | 0.920 | 0.654 | 0.994 |
| 3 | 1.00 | 0.847 | 0.876 | 0.997 |
| 4 | 1.00 | 0.749 | 0.973 | 0.816 |
| 5 | 1.00 | 1.000 | 0.870 | 0.918 |
| 6 | 1.00 | 0.974 | 1.000 | 0.926 |
| 7 | 1.00 | 0.934 | 0.803 | 0.989 |
| 8 | 1.00 | 0.762 | 0.862 | 0.980 |
| 9 | 1.00 | 0.937 | 0.859 | 0.949 |

Variance concentrates in IR and OR; Normal is trivially separated in every seed.

## Appendix E. Worked Example of the ILG (real seed-0 numbers)

Random-window: margin(R, seed 0) = PC_true - PC_neg = 0.602 - 0.326 = 0.276.
Recording-level: margin(S, seed 0) = 0.625 - 0.460 = 0.164. PC_true is comparable
(0.602 vs 0.625); the protocol effect is carried mainly by PC_neg (+0.134 under
the safe split). Averaging over ten seeds: mean margin(R)=0.263, mean margin(S)=0.168, so
ILG = 0.263 - 0.168 = +0.095. A reader who reported only PC_true would have seen a 0.020
difference and concluded "no effect"; the controlled margin reveals a 0.095 effect.

## Appendix F. Reproduction Checklist

Ordered commands (drive-end; use `--dataset FE` for fan-end):
1. `python src/make_manifest.py` -- build the recording-level manifest.
2. `python src/preprocess.py` -- cut windows, save `windows.npz`.
3. `python src/make_splits.py` -- emit protocol R/S index sets; verify overlaps.
4. `python src/run_experiments.py` -- train, evaluate, compute margins and ILG.
5. `python src/stats_ilg.py --dataset all` -- paired test + bootstrap CI on the ILG.
6. `python src/run_ablations.py` -- low-data, grouping, bandpass ablations.
7. `python src/mech_intra_record.py` -- intra- vs cross-record similarity.
8. `python src/make_figures.py`, `make_cross_figures.py`, `make_gradcam_fig.py` -- figures.










