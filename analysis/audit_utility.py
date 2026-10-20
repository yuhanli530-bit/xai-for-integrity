#!/usr/bin/env python3
"""Compare explanation-assisted auditing against decision-only review.

Input CSV columns: source_group,case_id,condition,is_error,alarm_score,status.
Thresholds must be frozen on development data before this held-out CSV is read.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def truth(value):
    value = value.strip().lower()
    if value in ("1", "true", "yes"):
        return True
    if value in ("0", "false", "no"):
        return False
    raise ValueError("Expected boolean label, got " + value)


def rates(rows, threshold):
    errors = [r for r in rows if r["is_error"]]
    clean = [r for r in rows if not r["is_error"]]
    return {
        "error_recall": sum(r["alarm_score"] >= threshold for r in errors) / len(errors) if errors else None,
        "false_alarm_rate": sum(r["alarm_score"] >= threshold for r in clean) / len(clean) if clean else None,
        "error_n": len(errors), "clean_n": len(clean),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--decision-only-threshold", type=float, required=True)
    p.add_argument("--with-explanation-threshold", type=float, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with args.csv.open(newline="") as f:
        raw = list(csv.DictReader(f))
    rows, failed = [], []
    for r in raw:
        if r.get("status") != "complete":
            failed.append(r)
            continue
        score = float(r["alarm_score"])
        if not 0 <= score <= 1:
            raise SystemExit("alarm_score must be in [0,1]")
        rows.append({"source_group": r["source_group"], "case_id": r["case_id"],
                     "condition": r["condition"], "is_error": truth(r["is_error"]), "alarm_score": score})
    indexed = {}
    for r in rows:
        key = (r["source_group"], r["case_id"], r["condition"])
        if key in indexed:
            raise SystemExit("Duplicate source/case/condition: " + str(key))
        indexed[key] = r
    thresholds = {"decision_only": args.decision_only_threshold,
                  "with_explanation": args.with_explanation_threshold}
    conditions = {c: [r for r in rows if r["condition"] == c] for c in thresholds}
    group_values = defaultdict(dict)
    for group, case, condition in indexed:
        group_values[group].setdefault(case, {})[condition] = indexed[(group, case, condition)]
    paired_groups = []
    for group, cases in group_values.items():
        diffs = []
        for case, pair in cases.items():
            if set(pair) != set(thresholds):
                continue
            if pair["decision_only"]["is_error"] != pair["with_explanation"]["is_error"]:
                raise SystemExit("Truth labels differ between paired reviewer conditions")
            if pair["decision_only"]["is_error"]:
                d = int(pair["with_explanation"]["alarm_score"] >= thresholds["with_explanation"]) - int(pair["decision_only"]["alarm_score"] >= thresholds["decision_only"])
                diffs.append(d)
        if diffs:
            paired_groups.append(sum(diffs) / len(diffs))
    rng = random.Random(args.seed)
    boots = []
    if paired_groups:
        for _ in range(args.bootstrap):
            boots.append(sum(paired_groups[rng.randrange(len(paired_groups))] for __ in paired_groups) / len(paired_groups))
        boots.sort()
    result = {
        "schema": "audit-utility-analysis-v1",
        "status": "complete" if not failed else "incomplete_denominator",
        "thresholds_frozen_on_development": thresholds,
        "rows_planned": len(raw), "rows_complete": len(rows), "rows_failed_or_missing": len(failed),
        "condition_metrics": {c: rates(conditions[c], thresholds[c]) for c in thresholds},
        "paired_error_discovery": {
            "estimand": "source-clustered difference in binary error discoveries, with explanation minus decision-only",
            "source_groups": len(paired_groups),
            "mean_delta": sum(paired_groups) / len(paired_groups) if paired_groups else None,
            "ci95_percentile": [boots[int(.025 * args.bootstrap)], boots[min(args.bootstrap - 1, int(.975 * args.bootstrap))]] if boots else None,
            "bootstrap_draws": args.bootstrap,
        },
        "interpretation_limit": "Thresholds must be set on separate development data. This does not establish utility from a single reviewer or unblinded annotations.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "rows_complete": len(rows), "source_groups": len(paired_groups), "output": str(args.output)}, indent=2))
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
