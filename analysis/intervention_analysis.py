#!/usr/bin/env python3
"""Analyze predeclared blinded claim-assertion triplets (schema v2)."""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

CONDITIONS = ('intact', 'referent_occlusion', 'matched_control_occlusion')
KEY_FIELDS = ('source_group', 'image_id', 'claim_id')
VALIDITY_FIELDS = ('intact_supported', 'referent_supported', 'control_supported')
METRICS = ('intact_assertion', 'referent_withdrawal', 'control_retention',
           'joint_evidence_pattern', 'selective_assertion_drop')


def read_csv(path, required):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        return list(reader)


def required_text(row, field):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Missing or invalid {field}')
    return value.strip()


def key_of(row):
    return tuple(required_text(row, k) for k in KEY_FIELDS)


def boolean(value):
    value = str(value).strip().lower()
    if value not in ('true', 'false', '1', '0'):
        raise ValueError(f'Expected boolean, got {value!r}')
    return value in ('true', '1')


def analyze(manifest, annotations, arm='G', bootstrap=10000, seed=20261021):
    if bootstrap < 1:
        raise ValueError('bootstrap must be positive')
    if not manifest:
        raise ValueError('The independent planned claim manifest must not be empty')
    planned = {}
    for row in manifest:
        key = key_of(row)
        if key in planned:
            raise ValueError(f'Duplicate planned claim: {key}')
        required_text(row, 'claim_text')
        planned[key] = row
    observed = defaultdict(dict)
    for row in annotations:
        if row.get('arm') != arm:
            continue
        key = key_of(row)
        if key not in planned:
            raise ValueError(f'Unplanned claim in selected arm: {key}')
        condition = row.get('intervention')
        if condition not in CONDITIONS:
            raise ValueError(f'Unknown intervention: {condition}')
        if condition in observed[key]:
            raise ValueError(f'Duplicate annotation: {key}/{condition}')
        observed[key][condition] = row

    issues, excluded, claim_values = [], [], []
    eligible_count = complete_triplets = 0
    validity_counts = Counter()
    for key, plan in planned.items():
        validity = None
        if plan.get('validity_status') == 'complete':
            try:
                required_text(plan, 'validity_annotator_id')
                validity = tuple(boolean(plan.get(f, '')) for f in VALIDITY_FIELDS)
            except ValueError as error:
                issues.append({'key': key, 'issue': f'invalid_input_validity: {error}'})
        else:
            issues.append({'key': key, 'issue': 'input_validity_not_complete'})
        eligible = validity == (True, False, True)
        eligible_count += eligible
        if validity is not None:
            for field, value in zip(VALIDITY_FIELDS, validity):
                validity_counts[field] += value
            if not eligible:
                excluded.append({'key': key, 'input_support_pattern': list(validity)})
        values = {}
        for condition in CONDITIONS:
            row = observed[key].get(condition)
            if row is None:
                issues.append({'key': key, 'condition': condition, 'issue': 'missing_row'})
                continue
            if row.get('status') != 'complete':
                issues.append({'key': key, 'condition': condition,
                               'issue': 'row_not_complete', 'status': row.get('status')})
                continue
            try:
                required_text(row, 'annotator_id')
                # Empty text can be a valid abstention; a missing field is not.
                if not isinstance(row.get('output_text'), str):
                    raise ValueError('Missing or invalid output_text')
                asserted = boolean(row.get('claim_asserted', ''))
                if asserted and not row['output_text'].strip():
                    raise ValueError('Empty output cannot assert a claim')
                values[condition] = asserted
            except ValueError as error:
                issues.append({'key': key, 'condition': condition,
                               'issue': f'invalid_output_annotation: {error}'})
        if len(values) != len(CONDITIONS):
            continue
        complete_triplets += 1
        if not eligible:
            continue
        intact, referent, control = (int(values[c]) for c in CONDITIONS)
        claim_values.append((key[0], {
            'intact_assertion': intact, 'referent_withdrawal': 1-referent,
            'control_retention': control,
            'joint_evidence_pattern': intact * (1-referent) * control,
            'selective_assertion_drop': (intact-referent) - (intact-control),
        }))

    grouped = defaultdict(list)
    for source, values in claim_values:
        grouped[source].append(values)
    clusters = [{m: mean(v[m] for v in values) for m in METRICS}
                for _, values in sorted(grouped.items())]
    samples = {m: [] for m in METRICS}
    if len(clusters) >= 2:
        rng = random.Random(seed)
        for _ in range(bootstrap):
            selected = [clusters[rng.randrange(len(clusters))] for _ in clusters]
            for metric in METRICS:
                samples[metric].append(mean(x[metric] for x in selected))
    estimates = {}
    for metric in METRICS:
        draws = sorted(samples[metric])
        estimates[metric] = {
            'source_macro_mean': mean(c[metric] for c in clusters) if clusters else None,
            'claim_macro_mean': mean(v[metric] for _, v in claim_values) if claim_values else None,
            'source_cluster_ci95': [draws[int(.025*bootstrap)],
                                   draws[min(bootstrap-1, int(.975*bootstrap))]] if draws else None,
        }
    status = 'incomplete_denominator' if issues else ('complete' if claim_values else 'no_eligible_claims')
    return {
        'schema': 'intervention-analysis-v2', 'arm': arm, 'status': status,
        'planned_claims': len(planned), 'planned_rows': len(planned)*len(CONDITIONS),
        'observed_rows_selected_arm': sum(len(x) for x in observed.values()),
        'complete_output_triplets': complete_triplets,
        'eligible_planned_claims': eligible_count, 'analyzed_claims': len(claim_values),
        'source_groups': len(clusters), 'input_validity_counts': dict(validity_counts),
        'excluded_by_input_validity': excluded, 'issues': issues,
        'primary_metric': 'joint_evidence_pattern',
        'estimand': 'Equal source weights; equal eligible-claim weights within each source. '
                    'Primary pattern asserts the fixed claim on intact/control inputs and withdraws it on referent occlusion.',
        'estimates': estimates if status == 'complete' else None,
        'complete_case_diagnostics': estimates if status != 'complete' else None,
        'bootstrap': {'draws': bootstrap, 'seed': seed, 'unit': 'source_group',
                      'interval_status': 'available' if len(clusters) >= 2 else 'insufficient_source_groups'},
        'interpretation_limit': 'Requires input-only validity labels frozen before output review, blinded qualified output labels, '
            'and a predeclared independent holdout. Occlusion sensitivity is evidence-dependence under this intervention, '
            'not proof of explanation truth, natural-image causality, auditor utility, or training-seed stability. '
            'A positive selective drop alone is insufficient; inspect the joint pattern and all component rates.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path, help='V2 blinded output annotations')
    parser.add_argument('--manifest', type=Path, required=True, help='Independent planned claims and input-only validity labels')
    parser.add_argument('--arm', default='G')
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261021, help='Bootstrap RNG seed, not training seed')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = read_csv(args.manifest, (*KEY_FIELDS, 'claim_text', 'validity_status',
                                     'validity_annotator_id', *VALIDITY_FIELDS))
        rows = read_csv(args.csv, (*KEY_FIELDS, 'arm', 'intervention', 'status', 'output_text',
                                  'claim_asserted', 'annotator_id'))
        result = analyze(plan, rows, args.arm, args.bootstrap, args.seed)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'planned_claims', 'analyzed_claims', 'source_groups')}, indent=2))
    if result['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
