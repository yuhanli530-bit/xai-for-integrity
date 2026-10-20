# Two method lines for XAI for Integrity

## Aim

The project asks whether an explanation is tied to the visual evidence it names, and whether that tie helps a separate auditor find incorrect forensic reports. Localization, textual similarity, causal dependence, and audit usefulness are distinct outcomes. Neither an SAE feature label nor a segmentation score establishes that a claim is true.

## Line A — spatial grounding (G/P)

**Mechanism.** Add a region loss to the visual prediction states aligned with complete, annotated explanation spans. G uses the native image representation and explicit referent masks. P is the proposed gradient-routing variant and is not ready for a full run until its routing is implemented and checked on the actual model.

**Primary question.** Does direct mask supervision improve artifact localization, and does any localization gain transfer to claim-level faithfulness under controlled visual interventions?

**Key controls.** Native baseline; G; P only after the gradient check; matched initialization, data order, update count, and final-checkpoint policy; measured decoder calls, wall time, and memory. Include a compute-matched control before making equal-compute claims.

**Current evidence.** G has one exploratory seed-42 result on 1,000 images. The test set has already informed iteration, so it cannot serve as confirmatory evidence. The result supports only a pilot localization observation; it does not establish faithful text or improved audit decisions.

## Line B — sparse causal evidence concepts (SAE-Causal)

**Motivation.** A pixel/region loss says *where* a referent is, but does not expose which visual representation drives a claim. Vision SAEs provide a way to decompose patch activations into sparse features, and prior work has shown that CLIP-vision SAE features can be intervened on to affect downstream VLM outputs. PatchSAE work also motivates retaining per-patch spatial structure. Those results make the tool plausible; they do not establish forensic usefulness or transfer to LEGION/GLaMM.

**Proposed method.** Freeze the trained forensic VLM and its exact vision encoder. Fit or adapt a sparse autoencoder to patch activations from that encoder on training images only. For each atomic explanation claim, use targeted activation patching on the corresponding claim-token log-probability to find SAE features whose removal or resampling changes the claim. Aggregate the spatially located feature activations into an evidence map. A claim is eligible for the evidence-conditioned decoder only when it has a sufficiently stable, causally tested feature set; otherwise the model abstains or marks it unsupported. The trainable component is a small claim-to-feature selector/gate, not a replacement of the base model. Do not use generated feature labels as ground-truth concepts.

**Objective and outputs.** Start with a post-hoc feasibility stage. If it passes, train the selector with the native text objective plus (i) a sparsity/complexity penalty on selected features and (ii) a counterfactual consistency term that reduces support for a claim when its causal evidence features are ablated, while leaving matched non-evidence features comparatively unchanged. Produce per-claim feature IDs, patch support maps, causal effect sizes, and abstention reasons. Freeze feature dictionary and thresholds before the held-out evaluation.

**Why this is XAI.** It exposes internal visual features, relates them to individual explanation claims, and tests the relationship by interventions. A colored activation map alone is not the explanation or the faithfulness test; the causal patching result is required.

## Line B gates and experiments

1. **B0, compatibility.** Identify the exact CLIP/ViT layer and tensor used by the existing model. Verify shape, preprocessing, patch-to-image mapping, and unchanged native outputs when the SAE path is disabled. Reject a pretrained SAE if its encoder revision/layer/normalization does not match.
2. **B1, dictionary feasibility.** Train or adapt the SAE using train-only images. Report reconstruction error, L0 activity, dead features, feature stability across seeds, and qualitative examples reviewed blind to test labels. Compare TopK/BatchTopK against a linear sparse coding or dense-activation baseline.
3. **B2, causal feature audit.** On train/validation only, patch candidate features and matched random features. Measure claim-token log-probability changes, generated claim changes, and spatial overlap with the referent. Test reconstruction-preserving ablation/resampling as well as zero ablation; zeroing alone can create off-manifold states.
4. **B3, method comparison.** Compare native baseline, post-hoc SAE explanation, and gated SAE-Causal. Keep G/P as a separate method line. After both lines pass independently, use a small factorial comparison (baseline, G, SAE-Causal, G+SAE-Causal) to test complementarity.
5. **B4, held-out claims.** Evaluate faithfulness with blinded claim-support labels and referent-vs-matched-control visual interventions. Report false claims, unsupported-claim abstentions, per-claim causal dependence, localization, and audit error-detection at a fixed false-alarm rate. Use a new source-disjoint test split; do not reuse the exploratory SynthScars test set.

**Stop conditions.** Stop if exact vision activations cannot be hooked without changing native predictions, reconstruction destroys relevant claim behavior, candidate features are unstable, or SAE feature interventions are no more specific than matched random-feature interventions. In that case retain Line A and use pixel/region interventions for faithfulness; do not force an SAE result.

## Compute and feasibility boundary

This is a staged line, not a promise that the full SAE pipeline is already implemented. The first useful milestone is a small feature-dictionary and causal-patching pilot on train/validation images, which needs no full VLM retraining. Only proceed to selector training and a large held-out run if B0–B2 pass. Estimate storage and GPU time from actual activation dimensions and sampled train images before reserving a large job.

## Selected references

- Pach et al., [Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models](https://arxiv.org/abs/2504.02821), NeurIPS 2025; [code](https://github.com/ExplainableML/sae-for-vlm). Demonstrates feature decomposition and interventions in CLIP vision encoders, including downstream VLM steering.
- Stevens et al., [Sparse Autoencoders for Scientifically Rigorous Interpretation of Vision Models](https://arxiv.org/abs/2502.06755), 2025; [SAE-V project](https://osu-nlp-group.github.io/SAE-V/). Emphasizes testing feature hypotheses with interventions, not feature visualization alone.
- [PatchSAE: Sparse Autoencoders for Vision Transformers with Patch-Wise Spatial Attributions](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3d5b603d631d595f56bc36b373458b27-Abstract-Conference.html), ICLR 2025. Motivates spatially resolved feature attribution in ViTs.

## Claims not yet supported

Do not claim that SAE-Causal improves accuracy, faithfulness, or auditor performance until the experiments above are completed on untouched held-out source groups with independent labels. SAE features are a hypothesis-generating and intervention unit; they are not automatically human concepts, causal explanations, or proof of image authenticity.
