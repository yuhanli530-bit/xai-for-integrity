# XAI for Integrity

Research code for testing whether forensic explanations are grounded in image regions, causally depend on the evidence they name, and help an independent reviewer detect errors.

The project now tracks two distinct method lines: supervised spatial grounding (G/P) and SAE-based sparse causal evidence concepts. The latter is a staged research proposal with feasibility gates, not implemented or validated code yet. See [`docs/METHOD_LINES.md`](docs/METHOD_LINES.md) for the method and literature rationale.

## Current evidence

The included G method adds an auxiliary region loss for the prediction states that produce each complete, annotated explanation. The existing single-seed pilot (seed 42, 1,000 SynthScars test images) improved localization metrics but did not improve text-similarity metrics. It is an exploratory result: one seed, a test set already used during iteration, no direct explanation-faithfulness result, and extra mask-decoder computation/supervision. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the complete claim boundary and next experiments.

## Repository layout

- `src/xai_integrity/reason_grounding.py`: G auxiliary-loss controller, adapted from the frozen research implementation.
- `src/xai_integrity/explanation_data.py`: strict training-example, explanation-span, and mask provenance joins.
- `src/xai_integrity/train_native.py`: finite native LEGION training loop and receipts.
- `scripts/launch_native.py`, `scripts/evaluate_official.py`: finite train/evaluation launchers.
- `analysis/`: frozen-result collection and training-log audits, plus generic paired bootstrap and intervention analysis.
- `scripts/build_interventions.py`: intact/referent/control triplets with exactly equal-area, same-shape disjoint masks. See [`docs/INTERVENTIONS.md`](docs/INTERVENTIONS.md) for the v2 input-validity and blinded output-annotation protocol.
- `configs/experiment_matrix.yaml`: planned G/P and SAE-Causal experiments. It does not launch work by itself.
- `TODO.md`: staged forward research plan, including the 3-run, single-seed training batch and gated SAE-Causal, external-domain, intervention, and auditor-utility studies.

Model weights, SynthScars, generated outputs, credentials, and GPU-server artifacts are intentionally external. The repository does not bundle them.

## Reproduction prerequisites

This code targets the pinned LEGION/GLaMM training stack used by the experiment, including its prepared source tree, exact pretrained assets, and official SynthScars split. It is not a one-command public reproduction: access to the lab's prepared model bundle and a compatible four-GPU host is required. `docs/REPRODUCE.md` describes the required mount points, preflight, launch, evaluation, and analysis steps. The historical exact seed-42 run is preserved as a protocol; the next single-seed and intervention experiments are separately identified as planned work.

## Quick start

```bash
git clone <repository-url>
cd xai-for-integrity
pip install -r requirements-analysis.txt
python analysis/paired_bootstrap.py --help
python analysis/intervention_analysis.py --help
python analysis/audit_utility.py --help
```

Large jobs must be launched only after reviewing the frozen protocol and ensuring the GPU allocation is exclusive. Training launchers never terminate unrelated processes or automatically retry a failed run.

## Citation and upstream project

The model and dataset are based on [LEGION](https://github.com/opendatalab/LEGION) and [SynthScars](https://huggingface.co/datasets/khr0516/SynthScars). Follow their licensing and access terms. This repository contains the research adaptation and analysis tools, not a copy of their full source distribution or assets.

## Status

The code in this repository is an experiment handoff. The current single-seed G result is exploratory; P, SAE-Causal, and all new generalization/faithfulness runs remain pending until implemented where needed, launched, and independently reviewed.
