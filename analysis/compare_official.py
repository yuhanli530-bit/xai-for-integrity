#!/usr/bin/env python3
"""Finite CPU-only G/baseline result collector; no model/test-image loading.

Reads completed official score/receipts only. Keeps every primary metric and
failure, never selects checkpoints/runs, starts jobs, retries, or creates cron.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import time

FORMAL = Path(os.environ.get('XAI_FORMAL_ROOT', Path(__file__).resolve().parents[1]))
ROOT = FORMAL / 'official_reason_grounding_20261010'
BASELINE = FORMAL / 'c_final_eval_v1/S42_OFFICIAL_SCORE.json'
BASELINE_SHA = '74059e0d6c7df798f043ce35041a0d9a286b1b187c4cae117fa8de7ffe751aef'
BASELINE_CHECKPOINT = '298a8fe6c52eca941aa18b00e6e35b15b918a332d253347eb5d392524dcce8d4'
EVAL_SNAPSHOT_SHA = '088e70c761f5609a64301bd3c9566d8157564835f1481c9eceb1a732ce318c68'
TRAIN_SNAPSHOT_SHA = 'c18e801c5605f9d76bb2d3409d93f1f246c820476d7d2a4d82a1db31f0359d70'
WORKER_SHA = 'd711e3f3bf3b15e4fcc09320a5d64e52828d628a90961a69a35711f99ebf2ebc'
OBJECTIVE_SHA = '6bfa73ac681b7c9527153264ffb44044238e183570e12e7d7e937a404361f799'
SPEC_SHA = 'ed727a2d5b00b232d9cec1680a755722dc8b9aafafb71b6099f0ade4a13b65b3'
INITIAL_SHA = '89fba3d27b25d6942c8bbebff6f4f05a523d20b1fede27051e3bfaec8bc28f61'
ORDER_SHA = '674636c88a12e0fc419e0dd9fce7bd5eb2efc3b3109a77e4f968428cb093522e'
TEST_SHA = '3bb55380f9ad75d76c26ad9a18035296be4ab2a7255b5e2fa11b131a2417e7e4'
SCORER_PINS = {
    'formal_legion_20261007/c_final_eval_v1/score_official_with_provenance.py': '4a2c439368fc03169884516520743634182ebc809d00c5aa3e5bd0147325d436',
    'official_legion_20261006/metrics.py': '5bab86f477922166f661624589f6d5dce1b8109d0650624e89a2656b9ddf8337',
    'official_legion_20261006/SUPPORT_ASSETS.json': '40e03385266f783c3054c91ee9a363ccaf52f337d7e29b7ad4da17e94ecf851b',
    'repos/LEGION/eval/utils.py': '496093ad23b6050c5006109ec1ef6a1ba31db00de0a9e9325338cd04dfd6e0fe',
}
METRICS = ('miou_two_classes', 'foreground_f1', 'rouge_l_f1', 'css')
LIMITS = ('Single seed42; one declared G method and the completed fixed baseline, no best-run selection. '
          'Native data, initialization, modules, optimizer, batch16 and2109 updates are matched. '
          'G adds explanation-state mask queries and repeated official mask supervision, so this is not equal training compute. '
          'Benchmark localization/text alignment does not establish statement truth, explanation faithfulness, calibrated confidence, or SOTA. '
          'G is an exploratory iteration after earlier TEST results; fixed final2109, no within-run TEST checkpoint selection.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def score_record(path, method, expected_sha=None):
    path = Path(path)
    actual_sha = sha(path)
    if expected_sha:
        require(actual_sha == expected_sha, 'Fixed baseline score bytes differ')
    data = read(path)
    require(data.get('status') == 'complete' and data.get('planned') == data.get('completed') == 1000,
            'Incomplete full score denominator')
    require(data.get('seed') == 42 and data.get('checkpoint_global_steps') == 2109 and data.get('method_id') == method,
            'Wrong fixed seed/final checkpoint/method score')
    rows = data.get('rows', [])
    names = [row['image'] for row in rows]
    require(len(names) == len(set(names)) == 1000 and digest(sorted(names)) == ORDER_SHA,
            'Different or incomplete official TEST1000 identities')
    raw, times = data['metrics_raw'], data['metrics_times_100']
    for key in METRICS:
        require(type(raw.get(key)) in (int, float) and math.isfinite(raw[key]) and
                type(times.get(key)) in (int, float) and math.isfinite(times[key]) and
                math.isclose(times[key], 100 * raw[key], rel_tol=1e-12, abs_tol=1e-12) and
                all(type(row.get(key)) in (int, float) and math.isfinite(row[key]) for row in rows) and
                math.isclose(raw[key], sum(row[key] for row in rows) / 1000, rel_tol=1e-12, abs_tol=1e-12),
                'Invalid metric values/scale/image-macro aggregation: ' + key)
    gt_identity = sorted((row['image'], row['annotation_id'], row['reference_text'],
                          row['counts']['tp'] + row['counts']['fn'],
                          row['counts']['tn'] + row['counts']['fp']) for row in rows)
    return {
        'status': 'complete', 'score_path': str(path), 'score_sha256': actual_sha,
        'method_id': method, 'denominator': 1000, 'planned': 1000, 'completed': 1000,
        'failed': 0, 'failure_count_basis': 'unique complete1000 rows; G additionally verifies all4 shard receipts',
        'checkpoint_sha256': data['checkpoint_sha256'],
        'metrics_raw': {key: raw[key] for key in METRICS},
        'metrics_times_100': {key: times[key] for key in METRICS},
        'image_names_sha256': digest(sorted(names)), 'scored_ground_truth_sha256': digest(gt_identity),
        'rouge_implementation': data['rouge_implementation'], 'css_model': data['css_model'],
        'aggregation': data['aggregation'], 'ranges': data['ranges'],
    }


def failed_shard_counts(queue_dir):
    records, completed, failed = [], 0, 0
    for index in range(4):
        path = queue_dir / ('shard_%d/RUN_RECEIPT.json' % index)
        if path.is_file():
            row = read(path)
            completed += row.get('raw_completed', 0)
            failed += row.get('raw_failed', 0)
            records.append({'path': str(path), 'sha256': sha(path), 'receipt': row})
    return {'completed': completed, 'failed': failed, 'denominator': 1000,
            'unattempted_or_unreported': 1000 - completed - failed,
            'partial_or_failed_shards': records, 'partial_metrics_compared': False}


def observe(baseline):
    train_path = ROOT / 'runs/G_main_s42_v1/STATUS.json'
    queue_path = ROOT / 'evaluations/G_main_s42_v1/STATUS.json'
    expected_run = train_path.parent / 'candidate_reason_grounding'
    out = {'candidate': 'G', 'status': 'pending', 'training_status_path': str(train_path),
           'queue_status_path': str(queue_path), 'denominator': 1000}
    train = read(train_path) if train_path.is_file() else None
    queue = read(queue_path) if queue_path.is_file() else None
    out.update(training_status=train.get('status') if train else 'missing',
               queue_status=queue.get('status') if queue else 'missing')
    if train:
        require(train.get('schema') == 'official_reason_grounding_launch_v1' and
                train.get('method_variant') == 'official_reason_grounding' and train.get('mode') == 'main' and
                train.get('seed') == 42 and train.get('coefficient') == .1 and train.get('gpus') == [0, 1, 3, 5] and
                train.get('source_snapshot_sha256') == TRAIN_SNAPSHOT_SHA and
                train.get('train_worker_sha256') == WORKER_SHA and train.get('reason_module_sha256') == OBJECTIVE_SHA,
                'Wrong G training identity/frozen source')
        if train.get('status') == 'failed':
            return dict(out, status='failed', failure_stage='training', failure=train,
                        **failed_shard_counts(queue_path.parent))
    if queue:
        require(queue.get('candidate') == 'G' and queue.get('snapshot_sha256') == EVAL_SNAPSHOT_SHA and
                Path(queue['training_run_dir']).resolve() == expected_run and
                Path(queue['training_status']).resolve() == train_path,
                'Wrong G queue/snapshot/training binding')
        if queue.get('status') == 'failed':
            return dict(out, status='failed', failure_stage='evaluation', failure=queue,
                        **failed_shard_counts(queue_path.parent))
    if not queue or queue.get('status') != 'full1000_official_score_complete':
        return out
    require(train and train.get('status') == 'full_complete', 'Score completed without training completion')
    score_path = queue_path.parent / 'score/OFFICIAL_SCORE.json'
    require(Path(queue['score']).resolve() == score_path, 'Queue score path changed')
    record = score_record(score_path, 'official-native-reason-grounding-seed42-final2109')
    for key in ('image_names_sha256', 'scored_ground_truth_sha256', 'rouge_implementation', 'css_model', 'aggregation', 'ranges'):
        require(record[key] == baseline[key], 'Different test identity/metric definition: ' + key)
    scorer = read(score_path.parent / 'SCORING_RECEIPT.json')
    require(scorer.get('status') == 'complete' and scorer.get('denominator') == 1000 and
            scorer.get('score_sha256') == record['score_sha256'] and scorer.get('source_sha256') == SCORER_PINS,
            'Final frozen scorer receipt mismatch')
    plan = read(score_path.parent / 'OFFICIAL_SCORE_PLAN.json')
    completion = read(queue_path.parent / 'TRAINING_COMPLETION.json')
    require(plan['training_completion'] == completion and
            completion['status'] == 'full_training_and_final_delta_verified' and
            Path(completion['training_run_dir']).resolve() == expected_run and
            Path(completion['launcher_status']).resolve() == train_path and
            completion['delta_weights_sha256'] == record['checkpoint_sha256'] and
            completion['global_steps'] == 2109 and completion['epoch'] == 3 and
            completion['coefficient'] == .1 and completion['reason_module_sha256'] == OBJECTIVE_SHA and
            completion['original_spec_sha256'] == SPEC_SHA and
            plan['baseline_scoring_sources_sha256'] == SCORER_PINS and
            plan['official_protocol']['test_annotation_sha256'] == TEST_SHA and
            plan['raw_output_policy'] == 'original union of all masks>0 and unchanged explanation; no quality gating',
            'Score plan/training/data/final-delta binding differs')
    shards, completed, failed, planned, oom_recovered = [], 0, 0, 0, 0
    for index in range(4):
        path = queue_path.parent / ('shard_%d/RUN_RECEIPT.json' % index)
        row = read(path)
        require(row.get('status') == 'complete' and row.get('raw_metrics_denominator') == 250 and
                row.get('planned') == row.get('raw_completed') == 250 and row.get('raw_failed') == 0 and
                row.get('parameters_unchanged') is True and row.get('sources_unchanged') is True and
                row['binding']['completion'] == completion and row.get('quality_gating') is False and
                row.get('ground_truth_inputs') is False and row.get('extra_model_pass') is False,
                'Incomplete, filtered, changed-model or incorrectly bound shard')
        require(set(row['memory_recovered_images']) == set(row['native_oom_images']) and
                not row['recovery_failed_images'] and not row['unattempted_images'], 'Invalid actual OOM/coverage history')
        completed += row['raw_completed']
        failed += row['raw_failed']
        planned += row['planned']
        oom_recovered += row['memory_recovered']
        shards.append({'path': str(path), 'sha256': sha(path), 'actual_native_oom_recovered': row['memory_recovered']})
    record.update(completed=completed, failed=failed, planned=planned,
                  actual_native_oom_recovered=oom_recovered, shard_receipts=shards,
                  scorer_receipt_sha256=sha(score_path.parent / 'SCORING_RECEIPT.json'),
                  score_plan_sha256=sha(score_path.parent / 'OFFICIAL_SCORE_PLAN.json'),
                  training_completion_sha256=sha(queue_path.parent / 'TRAINING_COMPLETION.json'),
                  delta_raw={key: record['metrics_raw'][key] - baseline['metrics_raw'][key] for key in METRICS},
                  delta_percentage_points={key: record['metrics_times_100'][key] - baseline['metrics_times_100'][key] for key in METRICS})
    return dict(out, **record)


def render(report):
    lines = ['# G full1000 baseline comparison', '', 'Status: `' + report['status'] + '`.', '',
             '| Run | Status | Completed / denominator | Failed | mIoU | F1 | ROUGE-L | CSS |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, row in [('LEGION baseline', report['baseline']), ('G reason grounding', report['candidates']['G'])]:
        values = [format(row['metrics_times_100'][key], '.8f') for key in METRICS] if row['status'] == 'complete' else ['—'] * 4
        lines.append('| ' + ' | '.join([name, row['status'], str(row.get('completed', '—')) + ' / 1000',
                                      str(row.get('failed', '—')), *values]) + ' |')
    lines += ['', 'Primary image-macro metrics are shown on the ×100 scale.', '',
              '| G minus baseline | Δ mIoU | Δ F1 | Δ ROUGE-L | Δ CSS |',
              '|---|---:|---:|---:|---:|']
    row = report['candidates']['G']
    values = [format(row['delta_percentage_points'][key], '+.8f') for key in METRICS] if row['status'] == 'complete' else ['—'] * 4
    lines += ['| G | ' + ' | '.join(values) + ' |', '', LIMITS, '', 'Score byte identities:']
    for name, row in [('baseline', report['baseline']), ('G', report['candidates']['G'])]:
        lines.append('- ' + name + ': `' + row.get('score_sha256', 'not available') + '`')
    if report['candidates']['G'].get('failure_stage'):
        lines += ['', 'Failure details and any partial shard counts are retained in COMPARISON.json; no partial metrics comparison.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=int, default=172800)
    parser.add_argument('--poll-seconds', type=int, default=15)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(0 <= args.wait_seconds <= 172800 and args.poll_seconds == 15, 'Finite maximum48h/fixed15second policy')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only collector requires explicitly hidden CUDA devices')
    baseline = score_record(BASELINE, 'LEGION-author-retrained-seed42-final-epoch3', BASELINE_SHA)
    require(baseline['checkpoint_sha256'] == BASELINE_CHECKPOINT, 'Fixed baseline checkpoint differs')
    current = observe(baseline)
    if not args.execute:
        print(json.dumps({'status': 'CPU_preflight_pass', 'baseline_sha256': baseline['score_sha256'],
                          'candidate_status': current['status'], 'gpu_started': False,
                          'wait_seconds': args.wait_seconds, 'poll_seconds': 15}))
        return
    require(not args.output_dir.exists(), 'Fresh summary output only; no overwrite or restart')
    args.output_dir.mkdir(parents=True)
    started = time.time()
    deadline = time.monotonic() + args.wait_seconds
    stopped = [False]
    def stop(signum, frame):
        stopped[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    source_sha = sha(__file__)
    state = {'status': 'waiting_results', 'pid': os.getpid(), 'hostname': socket.gethostname(),
             'started_unix': started, 'deadline_unix': started + args.wait_seconds,
             'poll_seconds': 15, 'source_sha256': source_sha, 'gpu_started': False,
             'new_training_jobs': 0, 'cron': False, 'external_notifications': False}
    try:
        while True:
            require(sha(__file__) == source_sha, 'Collector source changed during finite run')
            current = observe(baseline)
            if stopped[0]:
                terminal = 'interrupted'
            elif current['status'] == 'failed':
                terminal = 'terminal_failure'
            elif current['status'] == 'complete':
                terminal = 'complete'
            elif time.monotonic() >= deadline:
                terminal = 'deadline_exceeded'
            else:
                terminal = None
            state.update(status=terminal or 'waiting_results', observed_unix=time.time(), candidate_status=current['status'])
            atomic(args.output_dir / 'STATUS.json', state)
            if terminal:
                report = {'schema': 'official_G_full1000_comparison_v1', 'status': terminal,
                          'baseline': baseline, 'candidates': {'G': current}, 'observed_unix': time.time(),
                          'source_sha256': source_sha, 'limits': LIMITS, 'extra_compute_matched': False,
                          'claims': {'sota': False, 'calibration': False, 'report_truth': False, 'faithfulness': False},
                          'denominator_policy': 'fixed1000; failures retained; no partial/full score conflation'}
                atomic(args.output_dir / 'COMPARISON.json', report)
                (args.output_dir / 'COMPARISON.md').write_text(render(report))
                if terminal != 'complete':
                    atomic(args.output_dir / 'FAILURE_REPORT.json', report)
                break
            time.sleep(min(15, max(0, deadline - time.monotonic())))
    except BaseException as exc:
        state.update(status='collector_failed', error_type=type(exc).__name__, error=str(exc), ended_unix=time.time())
        atomic(args.output_dir / 'STATUS.json', state)
        atomic(args.output_dir / 'FAILURE_REPORT.json', dict(state, baseline=baseline, candidate=current))
        raise
    state['ended_unix'] = time.time()
    atomic(args.output_dir / 'STATUS.json', state)
    print(json.dumps({'status': state['status'], 'output_dir': str(args.output_dir)}))


if __name__ == '__main__':
    main()
