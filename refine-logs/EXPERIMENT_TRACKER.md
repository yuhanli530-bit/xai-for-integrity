# Experiment tracker

Last verified 2026-10-30. Goal: 推进到论文级别. No full objective completion claimed.

| Work item | State | Evidence / next gate |
|---|---|---|
| Report 2 | Revised and rendered | Course report, separate from scientific validation |
| Historical G result | Summary available; raw comparison not audited here | Existing docs report localization gains; remote read timed out during banner exchange, so process/run state is unverified |
| Historical H/S alternatives | Local comparison artifacts located | Retain negative alternatives; independent per-image reanalysis still needed |
| E3 mask construction | Implemented; engineering checks passed | `scripts/build_interventions.py`, `tests/test_interventions.py`: intact preservation, same shape/area, zero overlap, failure retention |
| E3 scoring | Implemented; engineering checks passed | Independent denominator, fixed claim, input-validity eligibility, joint directional pattern, source aggregation, missing-label rejection |
| Independent code review | Completed; initial findings fixed and rechecked | Reviewer confirmed closure; independently reran all 18 tests successfully |
| E3 real-model triplet inference | Pending | Needs the pinned frozen model adapter and identical generation settings |
| E3 actual dataset/labels | Pending | New source-disjoint manifest, image-only validity, blinded output annotation, fixed sample count |
| E3 paired multi-arm effect | Pending | Same complete claims/validity for all arms; paired source bootstrap needed |
| P implementation/full comparison | Pending | Real-model gradient gate and finite four-GPU training |
| Line B and combination | Pending | Exact encoder access, literature/feasibility evidence, mechanism and matched controls |
| External/negative controls | Pending | Licensed source-disjoint data and frozen protocol |
| Audit utility | Pending | Correct pairing/denominator audit before reviewer data analysis |
| Manuscript | Pending | Tables/figures must follow verified results; novelty/citations and claim audit required |

The prior goal turn made document progress. This turn changes authoritative experiment code and tests. Local engineering work continues despite the remote read failure; this is not a global blocked state. No running GPU job was verified and no training job was launched/restarted.
