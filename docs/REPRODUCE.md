# Reproduction handoff

## Assets and access

Keep these outside Git and mount them on the execution host:

1. A read-only clone of the pinned LEGION source revision and its required GLaMM/GLaMM-derived prepared source tree.
2. The exact pretrained GLaMM/SAM/CLIP assets and the prepared baseline training specification/parameter-contract receipts.
3. Official SynthScars train/test images and annotations, with original checksums.
4. A fresh output directory with enough space for logs and the final adapter/delta, plus exclusive access to four compatible GPUs.

The released upstream LEGION README documents its runtime and data layout. Do not commit weights, dataset images, annotation exports, predictions, or API credentials.

## Required path contract

The historical trainer expects an experiment root containing the prepared baseline bundle (`prepared_full_s42_v2`), original author source, `TRAINING_SPEC.json`, and the trainable-parameter witness. The run consumes the `train.json` file and checks exact image/conversation/mask joins before applying any auxiliary loss. The test split is not read by the trainer.

The currently copied historical code still contains lab-specific absolute paths and exact hash pins. Before another lab member launches it, set up the same mounted project layout or replace those constants with that member's verified asset manifest and have the reviewer re-audit the resulting bundle. Do not merely remove the hash checks. This is an explicit portability gate; the files are not advertised as portable until that gate is completed.

## Finite run order

1. Copy `.env.example` to a private host-only file and export the configured paths. Do not commit it.
2. Record upstream commit, model/checkpoint revisions, dataset checksums, CUDA/PyTorch/Transformers/DeepSpeed versions, GPU inventory, and free disk.
3. Run the launcher's CPU-only preflight and retain its receipt.
4. Run native zero-coefficient equivalence and real-model nonzero-gradient checks; retain receipts.
5. Run a short engineering benchmark on explicitly allocated GPUs. Never continue training from this checkpoint.
6. Launch one fresh full run into a new output directory. Use the frozen seed/update schedule. Do not start automatically after preflight.
7. After training completion, run the fixed full-denominator evaluation on the untouched split. Preserve all per-image outputs, failures, scorer version, and hashes.
8. Run paired image bootstrap and stratified error analysis. Use paired image-level uncertainty for the fixed seed; this does not measure training-seed variability.
9. For E3/E4, create new held-out source clusters and freeze the intervention/annotation protocol before inference.

The copied training entry point is a faithful historical seed-42 G bundle, not yet a configurable multi-arm orchestration CLI. Once all external paths and pinned assets are present, run `scripts/run_native_seed42.sh 0,1,3,5 /new/output/path`, then `scripts/run_eval_seed42.sh /new/output/path/candidate_reason_grounding /new/output/path/STATUS.json /new/eval/path 0,1,3,5`. The evaluator checks the exact terminal training receipt before reading test files. Both output directories must be new.

The matrix in `configs/experiment_matrix.yaml` is a plan; do not treat the single-seed baseline/G/P matrix as executable yet. P gradient routing requires implementation and review before launch.

## Results format

Use one row per image and arm. Required columns are `image_id`, `seed`, `arm`, `split`, `status`, `miou`, `foreground_f1`, `rouge_l_f1`, `css`, `empty_mask`, `zero_overlap`, `pixel_precision`, `pixel_recall`, and `source_group`. See `schemas/image_metrics.schema.json`.

For the planned fixed-seed comparison, run `python analysis/paired_bootstrap.py results/image_metrics.csv --model-seed 42 --arms G P --output results/paired_bootstrap.json`. The bootstrap resamples paired images; its intervals do not estimate training-seed variability.

For E3, use one row per image × claim × intervention × model and an independent planned-claim manifest. Follow `docs/INTERVENTIONS.md` and the v2 `schemas/interventions.schema.json` / `schemas/intervention_claims.schema.json`. V1 correctness labels require re-annotation; they cannot identify claim assertion. Keep raw inputs/outputs access-controlled; publish only permitted aggregates.
