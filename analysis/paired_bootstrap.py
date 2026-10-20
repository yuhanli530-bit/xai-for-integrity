#!/usr/bin/env python3
"""Strict paired bootstrap for complete per-image baseline/G/P metric tables.

CSV must follow schemas/image_metrics.schema.json. Missing/failed rows are
reported and cause a non-zero exit instead of silently changing denominators.
"""
import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("miou", "foreground_f1", "rouge_l_f1", "css")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--control", default="baseline")
    p.add_argument("--arms", nargs="+", default=["G", "P"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with args.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    issues = [r for r in rows if r.get("status") != "complete"]
    index = {}
    for row in rows:
        key = (row["seed"], row["image_id"], row["arm"])
        if key in index:
            raise SystemExit(f"Duplicate row: {key}")
        index[key] = row
    problems = []
    rng = random.Random(args.seed)
    output = {"schema": "paired-bootstrap-v1", "csv": str(args.csv), "bootstrap_draws": args.bootstrap,
              "resampling_unit": "image within each seed; seed summaries remain separate",
              "failed_or_missing_rows": len(issues), "comparisons": []}
    for seed in args.seeds:
        for arm in args.arms:
            keys = sorted({(s, image) for s, image, a in index if s == str(seed) and a == args.control} |
                          {(s, image) for s, image, a in index if s == str(seed) and a == arm})
            pairs = []
            for s, image in keys:
                left = index.get((s, image, args.control))
                right = index.get((s, image, arm))
                if left is None or right is None or left.get("status") != "complete" or right.get("status") != "complete":
                    problems.append({"seed": seed, "image_id": image, "arms": [args.control, arm], "issue": "unpaired_or_incomplete"})
                    continue
                pairs.append((left, right))
            result = {"seed": seed, "control": args.control, "arm": arm, "paired_n": len(pairs), "metrics": {}}
            for metric in METRICS:
                valid = []
                for left, right in pairs:
                    try:
                        x, y = float(left[metric]), float(right[metric])
                    except (TypeError, ValueError, KeyError):
                        problems.append({"seed": seed, "metric": metric, "issue": "missing_metric"})
                        continue
                    if not (math.isfinite(x) and math.isfinite(y)):
                        problems.append({"seed": seed, "metric": metric, "issue": "nonfinite_metric"})
                        continue
                    valid.append(y - x)
                if valid:
                    means = []
                    for _ in range(args.bootstrap):
                        means.append(statistics.mean(valid[rng.randrange(len(valid))] for __ in valid))
                    means.sort()
                    result["metrics"][metric] = {
                        "mean_delta": statistics.mean(valid),
                        "ci95_percentile": [means[int(.025 * args.bootstrap)], means[min(args.bootstrap - 1, int(.975 * args.bootstrap))]],
                        "positive_fraction": sum(v > 0 for v in valid) / len(valid),
                        "ties": sum(v == 0 for v in valid),
                    }
            output["comparisons"].append(result)
    output["pairing_issues"] = problems
    if issues or problems:
        output["status"] = "incomplete_denominator"
    else:
        output["status"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "comparisons": len(output["comparisons"]),
                      "failed_or_missing_rows": len(issues), "pairing_issues": len(problems), "output": str(args.output)}, indent=2))
    if output["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
