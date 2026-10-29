# Claim-level visual intervention protocol (v2)

This implements the existing E3 plan. It is an experimental measurement protocol, not a new empirical result. The current G localization result does not supply these labels. Do not treat the engineering fixtures as research data.

## Question and unit

For an atomic claim fixed independently of a method's outputs, does the model continue to assert it when its supporting visual evidence is present, and withdraw it when that evidence is removed? Compare the same claim under intact (I), referent occlusion (R), and matched unrelated-region occlusion (C). A source group must represent the actual shared source (for example a common original image or generation provenance); it is not an arbitrary batch number. Use the same planned claims and input images for every method.

Freeze claims from licensed reference annotations or independent human inspection before generating model responses. Keep the prompt identical across conditions; do not reveal the intervention or suggest the expected answer. Annotators judging outputs must not see method identity. Text similarity, another model's answer, and the model's confidence are not ground truth.

## Construct inputs

Create a CSV with columns `source_group,image_id,claim_id,claim_text,image_path,mask_path`. Paths resolve relative to this CSV. Each `(source_group,image_id,claim_id)` must be unique. The binary mask must already match the image's pixel dimensions and orientation; the generator never resizes it.

```bash
python scripts/build_interventions.py data/e3/claims.csv \
  --output-dir outputs/e3/inputs --control-attempts 300
```

The output directory must be fresh. Each claim gets three PNGs, their intervention masks, and three rows in `manifest.jsonl`. Intact PNGs preserve the decoded input pixels. The control is a translation of the entire referent mask, with exactly equal area and shape, no clipping, and zero referent overlap. Both edits use the same background-median fill. `occluded_pixels` counts the covered mask; `changed_pixels` counts actual pixel changes. Geometry does not guarantee a valid semantic intervention.

Candidate locations are sampled without replacement using the full claim identity as a deterministic seed. If the finite search cannot place a disjoint mask, all three rows are recorded as failed and the command exits 2. This does not prove that no valid position exists. Do not shrink the mask, relax the overlap rule, replace that claim after observing model responses, or drop its record. Empty manifests and duplicates fail before creating outputs. Masks covering the whole image cannot supply a disjoint control.

## Review input validity before output annotation

Copy `outputs/e3/inputs/validity_template.csv` to the study's planned manifest and complete it using image-only review. It contains every planned identity, including construction failures. Freeze these decisions before reviewing model responses, and use one manifest for every arm.

- `intact_supported`: the fixed atomic claim has visible support in I.
- `referent_supported`: it still has visible support in R, including evidence elsewhere in the image.
- `control_supported`: it has visible support in C.
- `validity_status`: complete, pending, or failed; `validity_annotator_id` identifies the reviewer/adjudication record.

The primary eligible set is the input-only pattern `(true, false, true)`. Other completed patterns are reported as exclusions, with their counts. Pending/failed reviews make the analysis incomplete. This filtering must not use the model's behavior or differ by arm. Record any adjudication policy and disagreements in the study protocol. Geometry alone must never auto-fill these labels.

## Label outputs

Run each frozen model on all three inputs and preserve generation settings and complete output text. Store one CSV row per planned claim × condition × arm, following `schemas/interventions.schema.json`:

`source_group,image_id,claim_id,arm,intervention,status,output_text,claim_asserted,annotator_id`

`claim_asserted` is true only if the response affirmatively makes the original atomic claim. Explicit denial, uncertainty, omission, or insufficient evidence is false. An empty but successfully generated response is a false assertion label, not a missing row; an inference failure is `status=failed`, not abstention. Other claims in the response require separate quality review and are not scored by this field. CSV booleans accept true/false or 1/0. The JSON schemas describe typed records; CSV import performs the required conversions and checks.

V1 `output_change_appropriate` annotations cannot be silently reused. They do not determine whether the original claim was asserted in each condition. Re-annotate original outputs using the v2 definition, retaining the old files for provenance.

## Estimands

For each eligible claim let `aI,aR,aC` be the binary assertion labels.

- **Primary joint evidence pattern:** `aI * (1 - aR) * aC`.
- Intact assertion rate: `aI`.
- Referent withdrawal rate: `1 - aR`.
- Control retention rate: `aC`.
- Selective assertion drop: `(aI - aR) - (aI - aC) = aC - aR`.

The ideal directional pattern `(1,0,1)` scores one on the primary metric. Always asserting, always abstaining, or withdrawing after any occlusion scores zero. Selectivity alone is insufficient: `(0,0,1)` has positive selectivity despite failing on the intact input. Report every component with the joint metric. Do not subtract referent-correctness and control-correctness rates: the ideal response is correct in both conditions and that subtraction gives zero.

The primary aggregation gives each source equal weight and each eligible claim within a source equal weight. Claim-macro values are secondary and explicitly named. Bootstrap entire source groups, keeping triplets together. The RNG seed is independent of model training seed 42. Intervals describe source sampling conditional on the frozen model, interventions, and labels; they do not quantify seed variability or annotation uncertainty. With fewer than two analyzed source groups, no interval is emitted.

```bash
python analysis/intervention_analysis.py data/e3/output_annotations.csv \
  --manifest data/e3/planned_claims_with_validity.csv --arm G \
  --bootstrap 10000 --seed 20261021 --output results/e3_G.json
```

Every planned claim requires all three output rows, even if input review makes it ineligible. Empty data, missing whole claims, failed generation, invalid labels, and missing intact rows cannot become a complete study. Incomplete output exits 2 and has `estimates=null`; available-case values are retained only under `complete_case_diagnostics`. Unexpected identities and duplicate annotations are hard errors. Other arms do not contaminate the selected arm. For method comparisons, use exactly the same completed manifest and a paired source bootstrap; separate per-arm intervals do not establish an improvement between arms. A dedicated paired multi-arm analysis is still required before that claim.

## Launch and interpretation boundary

The next small experiment uses validation data to check inference, mask placement, annotation agreement, and failure rates. It cannot become the paper's untouched test set. Freeze the large-study source list, sample count, exclusion/adjudication rules, prompts, selected methods, and stopping rule before held-out inference. Current prerequisites still missing: the independent claim manifest, blinded labels, and a validated triplet inference adapter for each frozen model.

Occlusions may cause distribution shifts or alter global image structure; include the matched controls, inspect input validity, and report exclusion/failure rates. A positive result measures selective evidence dependence under this intervention. It does not by itself demonstrate natural-image causal mechanisms, explanation truth, or human audit usefulness. The latter needs the separate E4 experiment. Preserve alternative methods and failed runs in the development record.

Engineering verification: `python -m unittest discover -s tests -v`.
