# Research TODO and idea backlog

This file separates established results, experiment plans, and exploratory ideas. A checkbox means the work has been completed and recorded; it does not mean a proposed method has already been validated.

## Immediate: make the current evidence reliable

- [ ] Freeze a clean train/validation/test split by source/scar family; record file manifests and hashes before model selection.
- [ ] Re-run the spatial-grounding (G) result across multiple seeds on the frozen split. Report localization, text quality, calibration, runtime, and memory as separate outcomes.
- [ ] Implement and run the P control with the same data, initialization, compute budget, and evaluation protocol as G.
- [ ] Re-run the untouched held-out evaluation once after all choices are frozen; keep exploratory/test-used results clearly marked.
- [ ] Add confidence intervals and paired tests for every planned comparison; publish per-example outputs and analysis receipts without bundling restricted images or weights.

## Method line A: supervised spatial grounding (G/P)

- [ ] Compare native baseline, G auxiliary region supervision, and the P control under matched compute and seeds.
- [ ] Test whether explanations point to the correct evidence using referent-region deletion, matched-control deletion, and insertion/recovery interventions.
- [ ] Measure whether region supervision helps an independent auditor detect fabricated or unsupported claims, not only whether masks overlap annotations.
- [ ] Stratify results by lesion/scar size, location, image quality, and source family; inspect failure cases blind to method identity.
- [ ] Run ablations for mask-loss weight, annotated-token coverage, and region decoder capacity only after the main comparison is frozen.

## Method line B: SAE-Causal evidence features

- [ ] B0 compatibility: identify the exact vision encoder layer/tensor and confirm patch indexing, preprocessing, and output invariance when the SAE path is disabled.
- [ ] B1 dictionary pilot: train/adapt an SAE on training images only; report reconstruction error, sparsity/L0, dead features, and feature stability across seeds.
- [ ] B2 causal specificity: patch/ablate candidate features and compare claim-token log-probability changes against matched random features and pixel/region controls.
- [ ] Build spatial evidence maps from patch-local SAE activations; test whether maps track claim referents on held-out validation examples.
- [ ] Add a small claim-to-feature selector/gate only if B0–B2 pass. Unsupported or unstable claims should abstain rather than receive a forced feature explanation.
- [ ] Compare native, post-hoc SAE, and gated SAE-Causal. Then run the pre-registered factorial comparison: baseline, G, SAE-Causal, and G+SAE-Causal.
- [ ] Stop the SAE line if encoder compatibility, feature stability, reconstruction fidelity, or intervention specificity fails. Keep that negative result and continue with line A.

## Integrity audit and generalization ideas

- [ ] Measure claim-level support/contradiction detection by an independent auditor with and without evidence maps; blind the auditor to model/method identity.
- [ ] Add counterfactual consistency: change or remove the named evidence and test whether the corresponding claim changes while unrelated claims remain stable.
- [ ] Test source/domain shift with source-family held-out splits and, if access permits, a second independently collected dataset.
- [ ] Audit shortcut reliance with background/context swaps, matched benign controls, and lesion/scar-preserving transformations.
- [ ] Report abstention coverage versus risk and calibration; include unsupported-claim rate and false reassurance rate.
- [ ] Keep localization, causal faithfulness, textual similarity, diagnostic utility, and auditor utility as distinct endpoints; do not collapse them into one score.

## Reproducibility and handoff

- [ ] Validate each large-job config against a frozen protocol and record code revision, dataset manifest, seed, hardware, wall time, and output checksums.
- [ ] Run small preflight jobs before multi-GPU runs; estimate activation storage and GPU hours from measured pilot dimensions.
- [ ] Save immutable per-example predictions, masks, intervention outcomes, and analysis tables under run IDs; do not overwrite prior results.
- [ ] Have a second researcher reproduce the frozen analysis from a clean checkout and compare result hashes/tables.
- [ ] Update `docs/EXPERIMENTS.md` and the claim ledger after every completed run; change this backlog only when decisions or evidence change.

## Exploratory ideas (not yet in the main claim set)

- [ ] Compare SAE feature dictionaries with simpler concept probes or linear sparse coding to test whether SAE complexity is necessary.
- [ ] Explore patch-level concept stability across augmentations and model seeds before assigning human-readable labels.
- [ ] Test whether causal feature groups reveal omitted evidence or contradictory evidence that region masks alone miss.
- [ ] Consider a reviewer-facing interface that shows claim, supporting patches, counterfactual effect, and uncertainty; evaluate it as a user study only after the offline audit task is stable.

