#!/usr/bin/env python3
"""Analyze blinded claim-level intervention annotations with source clustering."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

CONDITIONS = ("referent_occlusion", "matched_control_occlusion")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, help="Rows conforming to schemas/interventions.schema.json")
    p.add_argument("--arm", default="G")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with args.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_key = defaultdict(dict)
    for row in rows:
        key = (row["source_group"], row["image_id"], row["claim_id"], row["arm"])
        condition = row["intervention"]
        if condition in by_key[key]:
            raise SystemExit(f"Duplicate annotation for {key}/{condition}")
        by_key[key][condition] = row
    failed = [r for r in rows if r.get("status") != "complete"]
    complete_pairs = []
    missing = []
    for key, conditions in by_key.items():
        if key[-1] != args.arm:
            continue
        if all(c in conditions and conditions[c].get("status") == "complete" for c in CONDITIONS):
            complete_pairs.append((key, conditions))
        else:
            missing.append(key)
    denominators = defaultdict(lambda: {"n": 0, "appropriate": 0})
    for key, conditions in complete_pairs:
        for c in CONDITIONS:
            value = conditions[c].get("output_change_appropriate", "").strip().lower()
            if value not in ("true", "false", "1", "0"):
                missing.append({"key": key, "issue": f"invalid_label_{c}"})
                continue
            denominators[c]["n"] += 1
            denominators[c]["appropriate"] += value in ("true", "1")
    groups = defaultdict(lambda: {c: [] for c in CONDITIONS})
    for (group, _, _, _), conditions in complete_pairs:
        for c in CONDITIONS:
            val = conditions[c].get("output_change_appropriate", "").strip().lower()
            if val in ("true", "false", "1", "0"):
                groups[group][c].append(val in ("true", "1"))
    cluster_rows = []
    for group, values in groups.items():
        row = {c: (sum(values[c]) / len(values[c]) if values[c] else None) for c in CONDITIONS}
        if all(row[c] is not None for c in CONDITIONS):
            row["difference"] = row["referent_occlusion"] - row["matched_control_occlusion"]
            cluster_rows.append(row)
    rng = random.Random(args.seed)
    samples = []
    if cluster_rows:
        for _ in range(args.bootstrap):
            selected = [cluster_rows[rng.randrange(len(cluster_rows))] for __ in cluster_rows]
            samples.append(sum(r["difference"] for r in selected) / len(selected))
        samples.sort()
    result = {
        "schema": "intervention-analysis-v1",
        "arm": args.arm,
        "status": "complete" if not failed and not missing else "incomplete_denominator",
        "planned_rows": len(rows), "failed_or_missing_rows": len(failed),
        "complete_claim_pairs": len(complete_pairs), "source_groups": len(cluster_rows),
        "per_condition": {c: {"n": v["n"], "appropriate_rate": v["appropriate"] / v["n"] if v["n"] else None}
                          for c, v in denominators.items()},
        "source_clustered_difference": {
            "estimand": "appropriate-response rate after referent occlusion minus matched-control occlusion",
            "mean": sum(r["difference"] for r in cluster_rows) / len(cluster_rows) if cluster_rows else None,
            "ci95_percentile": [samples[int(.025 * args.bootstrap)], samples[min(args.bootstrap - 1, int(.975 * args.bootstrap))]] if samples else None,
            "bootstrap_draws": args.bootstrap,
        },
        "incomplete_pairs": [str(x) for x in missing],
        "interpretation_limit": "Requires blinded qualified annotations; this analysis does not infer faithfulness from text similarity alone.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "complete_claim_pairs": len(complete_pairs),
                      "source_groups": len(cluster_rows), "output": str(args.output)}, indent=2))
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
