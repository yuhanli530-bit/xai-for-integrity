# Experiment bridge update — 2026-10-30

## Implemented and checked

The existing E3 analysis subtracted two appropriate-response rates. An ideal response is appropriate for both referent and control occlusion, making that quantity zero. V2 instead measures the joint intact-assert / referent-withdraw / control-retain pattern and reports each component plus selective assertion drop. Input-only validity labels are separated from model-output annotations and fixed across arms.

The input generator now saves intact images and exact-area, same-shape disjoint controls. It rejects mismatched mask dimensions, prevents cross-image output collisions, and preserves failure rows. The analysis requires an independently planned manifest and will not report a complete estimate after an entire claim, an intact input, or a required label disappears. Source-macro and claim-macro estimates are explicit.

Independent review identified a missing-value loophole: None converted to text and empty outputs could be accepted as assertions. This was fixed with required-string checks, an empty-output consistency rule, and regression tests. The reviewer used the available model; the skill's specified cross-model tool was unavailable. Do not call this cross-model validation.

## Verification

`python -m unittest discover -s tests -v` passed 18 tests; the independent reviewer reran the same suite and confirmed all 18 pass. The final review reported no remaining substantive issue in the scoped implementation. Tests cover all eight binary response patterns, constant assertion/abstention, indiscriminate withdrawal, missing whole claims, omitted intact rows, invalid/null/short-CSV annotations, per-arm isolation, unequal source sizes, deterministic clustering, one-pixel/non-square masks, exact pixel preservation outside edits, path uniqueness, and unplaceable/misaligned masks.

These are engineering witnesses with synthetic fixtures. They provide no estimate of empirical explanation faithfulness, no new G performance result, and no trained-model comparison.

## External evidence access

A bounded, read-only SSH attempt to the configured historical training host failed during banner exchange. It did not verify whether the historical comparison process still exists. No job was started, stopped, retried, or inferred terminal from the timeout. Exact G score recovery remains pending; existing local H/S comparison artifacts are retained for the planned evidence inventory.

## Remaining gates

The real-model triplet inference adapter, independent held-out manifest, qualified labels, paired method comparison, P and Line B implementations, and all new large experiments remain pending. The research contract and tracker preserve the full paper objective and its evidence requirements.
