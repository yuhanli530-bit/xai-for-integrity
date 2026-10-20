#!/usr/bin/env python3
"""One CPU-only TRAIN log diagnostic; never imports Torch or reads TEST data.

Independent engineering helper; outside immutable training/evaluation snapshots.
Bounds append-only logs at the minimum completed rank update for a consistent
cross-rank analysis. Does not change the model, optimizer, loader, or live jobs.
"""
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

FORMAL = Path(os.environ.get('XAI_FORMAL_ROOT', Path(__file__).resolve().parents[1]))
ROOT = FORMAL / 'official_reason_grounding_20261010'
RUN = ROOT / 'runs/G_main_s42_v1/candidate_reason_grounding'
TRAIN_ANNOTATIONS = Path(os.environ.get('XAI_TRAIN_ANNOTATIONS', Path(os.environ.get('XAI_PROJECT_ROOT', Path(__file__).resolve().parents[1])) / 'data/synthscars/SynthScars/train/annotations/train.json'))
TRAIN_SHA = '94100c0b381f4b24745d75e5cba65872686ccf9e37af8e9fd669349e3509cd27'
WORKER_SHA = 'd711e3f3bf3b15e4fcc09320a5d64e52828d628a90961a69a35711f99ebf2ebc'
OBJECTIVE_SHA = '6bfa73ac681b7c9527153264ffb44044238e183570e12e7d7e937a404361f799'
SNAPSHOT_SHA = 'c18e801c5605f9d76bb2d3409d93f1f246c820476d7d2a4d82a1db31f0359d70'
INITIAL_SHA = '89fba3d27b25d6942c8bbebff6f4f05a523d20b1fede27051e3bfaec8bc28f61'
SPEC_SHA = 'ed727a2d5b00b232d9cec1680a755722dc8b9aafafb71b6099f0ade4a13b65b3'
API = 'prepare_native_forward_reason_grounding_one_extra_decoder_v1'


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), {'path': str(path), 'bytes_at_read': len(raw), 'sha256_at_read': sha_bytes(raw)}


def rows(path):
    raw = Path(path).read_bytes()
    lines = raw.splitlines()
    output, partial_last_line = [], False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            output.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not raw.endswith(b'\n'):
                partial_last_line = True
            else:
                raise
    return output, {'path': str(path), 'bytes_at_read': len(raw), 'sha256_at_read': sha_bytes(raw),
                    'partial_last_line_ignored': partial_last_line, 'valid_rows_at_read': len(output)}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def quantile(values, probability):
    values = sorted(values)
    index = (len(values) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def stats(values):
    if not values:
        return {'count': 0}
    good = [value for value in values if finite(value)]
    result = {'count': len(values), 'finite_count': len(good), 'nonfinite_count': len(values) - len(good)}
    if good:
        result.update(mean=statistics.mean(good), median=statistics.median(good),
                      min=min(good), max=max(good), p05=quantile(good, .05), p95=quantile(good, .95))
    return result


def loss_stats(events):
    keys = ('loss', 'ce_loss', 'mask_bce_loss', 'mask_dice_loss', 'mask_loss',
            'reason_bce_loss', 'reason_dice_loss', 'reason_grounding_loss')
    result = {key: stats([event['losses'][key] for event in events if key in event.get('losses', {})]) for key in keys}
    result['native_unweighted_BCE'] = stats([event['losses']['mask_bce_loss'] / .4 for event in events])
    result['native_unweighted_Dice'] = stats([event['losses']['mask_dice_loss'] / .2 for event in events])
    result['weighted_reason_over_native_mask_per_micro'] = stats([
        event['losses'].get('reason_grounding_loss', 0) / event['losses']['mask_loss']
        for event in events if event['losses']['mask_loss'] > 0])
    result['loss_scope'] = 'Per native microbatch; native mask terms already carry BCE=.4/Dice=.2; reason raw BCE/Dice reported only on eligible events, weighted reason=.1*(.4BCE+.2Dice).'
    return result


def diagnose():
    started = time.time()
    checks, errors, inputs, states = {}, [], [], {}
    for rank in range(4):
        state, metadata = read(RUN / 'receipts' / ('RUN_rank%d.json' % rank))
        states[rank] = state
        inputs.append(metadata)
    cutoff = min(state['optimizer_updates'] for state in states.values())
    low = max(1, cutoff - 199)
    annotation, metadata = read(TRAIN_ANNOTATIONS)
    inputs.append(metadata)
    if metadata['sha256_at_read'] != TRAIN_SHA:
        raise ValueError('Official TRAIN annotations changed')
    universe, nonempty, official_images = set(), set(), set()
    for row in annotation:
        aid, item = next(iter(row.items()))
        caption = item['caption'].strip('"').strip()
        official_images.add(item['img_file_name'])
        for index, ref in enumerate(item['refs']):
            if ref['sentence'] in caption:
                key = (aid, index)
                universe.add(key)
                if isinstance(ref.get('explanation'), str) and ref['explanation'].strip():
                    nonempty.add(key)
    all_events, first_events, last_events, all_updates, last_updates, last_timings = [], [], [], [], [], []
    coverage, images, native_images, histogram = set(), set(), set(), Counter()
    skips, exclusions, counter_sum, per_rank = Counter(), Counter(), Counter(), {}
    grad_errors, event_contract_errors, native_loss_nonfinite = [], [], []
    for rank in range(4):
        state = states[rank]
        fixed = {'rank': rank, 'mode': 'main', 'world_size': 4, 'seed': 42,
                 'effective_batch': 16, 'gradient_accumulation': 4, 'scheduled_updates': 2109,
                 'updates_requested': 2109, 'native_exposure_budget': 33744,
                 'coefficient': .1, 'max_expanded_length': 1536, 'extra_data': False,
                 'method_variant': 'official_reason_grounding', 'original_ds_config_unmodified': True,
                 'extra_LM_forwards': 0, 'extra_CLIP_forwards': 0, 'extra_SAM_encoder_forwards': 0}
        checks['rank%d_fixed_native_setting' % rank] = all(state.get(key) == value for key, value in fixed.items())
        checks['rank%d_frozen_sources_init' % rank] = (
            state.get('worker_sha256') == WORKER_SHA and state.get('reason_module_sha256') == OBJECTIVE_SHA and
            state.get('source_snapshot_sha256') == SNAPSHOT_SHA and state.get('original_spec_sha256') == SPEC_SHA and
            state.get('original_model_parameters_sha256') == INITIAL_SHA)
        before, after = state['trainable_contract_before'], state['trainable_contract_after']
        checks['rank%d_trainables_unchanged' % rank] = before == after and before['trainable_tensors'] == 297 and before['trainable_elements'] == 587427044
        p = RUN / 'receipts'
        logs = {}
        for name in ('REASON', 'UPDATES', 'TIMING', 'VISUAL_EXPOSURE'):
            log, metadata = rows(p / ('%s_rank%d.jsonl' % (name, rank)))
            inputs.append(metadata)
            logs[name] = log
        events = [row for row in logs['REASON'] if row['update'] <= cutoff]
        updates = [row for row in logs['UPDATES'] if row['update'] <= cutoff]
        timings = [row for row in logs['TIMING'] if row['update'] <= cutoff]
        exposures = [row for row in logs['VISUAL_EXPOSURE'] if row['micro'] < cutoff * 4]
        checks['rank%d_complete_ordered_common_prefix' % rank] = (
            len(events) == len(timings) == len(exposures) == cutoff * 4 and len(updates) == cutoff and
            [row['micro'] for row in events] == list(range(cutoff * 4)) and
            [row['update'] for row in updates] == list(range(1, cutoff + 1)))
        ref_root = FORMAL / 'official_explanation_response_20261010/runs/H_benchmark_s42_v1/candidate_H/receipts'
        ref_exposures, _ = rows(ref_root / ('VISUAL_EXPOSURE_rank%d.jsonl' % rank))
        ref_updates, _ = rows(ref_root / ('UPDATES_rank%d.jsonl' % rank))
        exposure_key = lambda row: (row['micro'], row['exposure_sha256'], row['image_paths'])
        checks['rank%d_first64_native_exposures_exact_H_short' % rank] = (
            [exposure_key(row) for row in exposures[:64]] == [exposure_key(row) for row in ref_exposures])
        checks['rank%d_first16_LRs_exact_H_short' % rank] = (
            [row['learning_rates'] for row in updates[:16]] == [row['learning_rates'] for row in ref_updates])
        rank_counters, rank_skips, rank_coverage = Counter(), Counter(), set()
        for event in events:
            receipt = event['receipt']
            if receipt.get('api') != API or receipt.get('coefficient') != .1 or any(receipt.get(key) != 0 for key in
                    ('extra_LM_forwards', 'extra_CLIP_forwards', 'extra_SAM_encoder_forwards', 'extra_backward_calls', 'extra_projection_forwards')):
                event_contract_errors.append({'rank': rank, 'micro': event['micro'], 'error': 'different objective/computation contract'})
            if receipt.get('completed'):
                count = receipt['additional_GT_mask_exposures']
                rank_counters.update(reason_events=1, reason_supervised_refs=count,
                                     extra_SAM_mask_decode_calls=receipt['extra_decoder_forwards'],
                                     extra_GT_pixel_exposures=receipt['additional_GT_pixel_exposures'])
                if not (receipt.get('native_exact_join') is True and receipt.get('auxiliary_rng_unchanged') is True and
                        receipt.get('extra_decoder_forwards') == receipt.get('extra_prompt_encoder_forwards') == 1 and
                        count == len(receipt['selected_refs']) == receipt['auxiliary_mask_count']):
                    event_contract_errors.append({'rank': rank, 'micro': event['micro'], 'error': 'binding/query/counter/RNG contract differs'})
                grad = receipt.get('auxiliary_query_gradient', {})
                if grad.get('finite') is not True or type(grad.get('nonzero')) is not bool or not finite(grad.get('max_abs')):
                    grad_errors.append({'rank': rank, 'micro': event['micro'], 'gradient': grad})
                for ref in receipt['selected_refs']:
                    key = (receipt['annotation_id'], ref['original_ref_index'])
                    coverage.add(key)
                    rank_coverage.add(key)
                    histogram[key] += 1
                images.add(Path(receipt['source']).name)
            else:
                rank_skips[receipt.get('skip_reason', '<unspecified>')] += 1
            for exclusion in receipt.get('excluded_refs', []):
                exclusions[exclusion['reason']] += 1
            if any(not finite(value) for value in event['losses'].values()):
                native_loss_nonfinite.append({'rank': rank, 'micro': event['micro'], 'losses': event['losses']})
        for row in exposures:
            native_images.update(Path(path).name for path in row['image_paths'])
        counter_sum.update(rank_counters)
        skips.update(rank_skips)
        # Own completed RUN may be ahead of the common cutoff; compare its own bounded prefix separately.
        own_counters = Counter()
        for event in logs['REASON']:
            if event['update'] <= state['optimizer_updates'] and event['receipt'].get('completed'):
                r = event['receipt']
                own_counters.update(reason_events=1, reason_supervised_refs=r['additional_GT_mask_exposures'],
                                    extra_SAM_mask_decode_calls=r['extra_decoder_forwards'],
                                    extra_GT_pixel_exposures=r['additional_GT_pixel_exposures'])
        checks['rank%d_completed_counters_match_RUN_snapshot' % rank] = all(state.get(key) == value for key, value in own_counters.items())
        aux_grad = state['reason_auxiliary_gradient']
        checks['rank%d_cumulative_aux_gradient_finite_nonzero' % rank] = aux_grad.get('finite') is True and aux_grad.get('nonzero_calls', 0) > 0
        checks['rank%d_total_module_gradients_finite' % rank] = all(g.get('finite') is True for g in state['total_module_gradient'].values())
        per_rank[str(rank)] = {'RUN_optimizer_updates': state['optimizer_updates'], 'author_validation_calls': state['author_validation_calls'],
                              'RUN_status': state['status'], 'common_prefix_counters': dict(rank_counters),
                              'common_prefix_distinct_refs': len(rank_coverage), 'skip_reasons': dict(rank_skips),
                              'cumulative_aux_gradient': aux_grad, 'cumulative_total_module_gradient': state['total_module_gradient']}
        all_events.extend(events)
        first_events.extend(row for row in events if row['update'] <= 200)
        last_events.extend(row for row in events if low <= row['update'] <= cutoff)
        all_updates.extend(updates)
        last_updates.extend(row for row in updates if low <= row['update'] <= cutoff)
        last_timings.extend(row for row in timings if low <= row['update'] <= cutoff)
    checks['all_events_fixed_contract'] = not event_contract_errors
    checks['all_losses_finite'] = not native_loss_nonfinite
    checks['all_aux_query_gradients_finite'] = not grad_errors
    checks['covered_refs_belong_to_official_TRAIN'] = coverage <= universe
    checks['covered_images_belong_to_official_TRAIN'] = images <= official_images
    last_complete = [row['receipt'] for row in last_events if row['receipt'].get('completed')]
    fresh_status, _ = read(ROOT / 'diagnostics/runs/real_grad_s42_v1/STATUS.json')
    fresh_rows, _ = rows(ROOT / 'diagnostics/runs/real_grad_s42_v1/SAMPLE_GRADIENTS.jsonl')
    fresh_evidence = []
    for row in fresh_rows:
        fresh_evidence.append({'micro': row['micro'], 'status': row['status'],
                               'weighted_reason_gradients': row['gradients']['weighted_reason'],
                               'vs_native_CE_LoRA_B': row['comparisons']['native_ce']['lora_B'],
                               'vs_native_mask': row['comparisons']['native_mask']})
    return {
        'schema': 'G_train_only_midrun_diagnostic_v1', 'observed_unix': started,
        'status': 'numeric_and_contract_checks_pass' if all(checks.values()) else 'diagnostic_issue',
        'test_metrics_or_images_or_annotations_read': False, 'gpu_started': False,
        'live_source_modified': False, 'run': str(RUN), 'common_completed_update_cutoff': cutoff,
        'recent_window': [low, cutoff], 'first_window': [1, min(200, cutoff)],
        'checks': checks, 'failed_checks': [key for key, passed in checks.items() if not passed],
        'event_contract_errors': event_contract_errors, 'aux_gradient_errors': grad_errors,
        'nonfinite_losses': native_loss_nonfinite, 'per_rank': per_rank,
        'coverage': {'official_refs': len(universe), 'official_nonempty_explanation_refs': len(nonempty),
                     'official_unique_images': len(official_images), 'aux_unique_refs': len(coverage),
                     'aux_nonempty_ref_coverage_fraction': len(coverage & nonempty) / len(nonempty),
                     'aux_unique_images': len(images), 'aux_image_coverage_fraction': len(images) / len(official_images),
                     'native_unique_images': len(native_images), 'exposure_count_distribution_per_seen_ref': stats(list(histogram.values())),
                     'all_common_prefix_counters': dict(counter_sum), 'whole_aux_skip_reasons': dict(skips),
                     'individual_ref_exclusion_reasons': dict(exclusions)},
        'first200_training_losses': loss_stats(first_events), 'recent200_training_losses': loss_stats(last_events),
        'recent200_gradient_norm': stats([row['global_gradient_norm'] for row in last_updates]),
        'recent200_aux_query_gradient_maxabs': stats([r['auxiliary_query_gradient']['max_abs'] for r in last_complete]),
        'recent200_aux_query_nonzero_fraction': sum(r['auxiliary_query_gradient']['nonzero'] for r in last_complete) / len(last_complete),
        'recent200_update_seconds_max_rank': stats([row['seconds_max_rank'] for row in last_updates]),
        'recent200_peak_allocated_bytes_max_rank': stats([row['peak_allocated_bytes_max_rank'] for row in last_updates]),
        'recent200_micro_forward_seconds': stats([row['native_plus_reason_forward_seconds'] for row in last_timings]),
        'recent200_micro_backward_seconds': stats([row['native_plus_reason_backward_seconds'] for row in last_timings]),
        'recent200_micro_whole_seconds': stats([row['whole_micro_seconds'] for row in last_timings]),
        'recent200_aux_decoder_seconds': stats([r['auxiliary_decoder_seconds'] for r in last_complete]),
        'fresh_initialization_aux_only_gradient_evidence': {
            'status': fresh_status['status'], 'valid_samples': fresh_status['valid_samples'],
            'scope': fresh_status['diagnosis_scope'], 'rows': fresh_evidence},
        'mechanism_boundary': [
            'Live pooled-query hooks are auxiliary-only and continuously nonzero, so the reason branch is not disconnected.',
            'Four earlier fresh-init TRAIN samples isolate nonzero auxiliary gradients into LoRA_B, shared projection and SAM. This rejects a decoder-only/disconnected path at those samples, not at all current updates.',
            'Live projection/SAM/LoRA hooks record TOTAL-loss gradients, not isolated reason gradients. They cannot identify how much current parameter change or benchmark gain is caused by reason supervision.',
            'Shared decoder can improve TRAIN mask loss without ensuring explanation hidden representations carry specific causal evidence. No current-state representation intervention was run.',
            'Teacher-forced history and referent GT remain; loss reduction/coverage do not demonstrate free-generated explanation faithfulness or authenticity/calibration.',
            'Different sampled images and learning rates between first/recent200 prevent interpreting loss trends as a causal method effect or held-out improvement.'],
        'input_train_log_snapshots': inputs,
    }


def render(report):
    first, recent = report['first200_training_losses'], report['recent200_training_losses']
    c = report['coverage']
    lines = ['# G TRAIN-only 中段诊断', '',
             '状态：`' + report['status'] + '`。共同完成前缀截至 update ' + str(report['common_completed_update_cutoff']) + '，最近窗口 ' + str(report['recent_window']) + '。',
             '', '未读取 G 的 TEST 图像、标注或指标；未改活跃源码、未启动 GPU。', '',
             '| 训练量（microbatch mean） | 前200步 | 最近200步 |', '|---|---:|---:|']
    for key in ('ce_loss', 'mask_bce_loss', 'mask_dice_loss', 'mask_loss', 'reason_bce_loss', 'reason_dice_loss', 'reason_grounding_loss'):
        lines.append('| ' + key + ' | ' + format(first[key]['mean'], '.6f') + ' | ' + format(recent[key]['mean'], '.6f') + ' |')
    lines += ['', '全累计辅助覆盖 ' + str(c['aux_unique_refs']) + '/' + str(c['official_nonempty_explanation_refs']) +
              ' 个非空解释 ref（' + format(c['aux_nonempty_ref_coverage_fraction'] * 100, '.2f') + '%），' +
              str(c['aux_unique_images']) + '/' + str(c['official_unique_images']) + ' 张图。',
              '', '整次辅助跳过原因：`' + json.dumps(c['whole_aux_skip_reasons'], ensure_ascii=False) + '`；单 ref 排除：`' +
              json.dumps(c['individual_ref_exclusion_reasons'], ensure_ascii=False) + '`。',
              '', '最近200步实际平均耗时 ' + format(report['recent200_update_seconds_max_rank']['mean'], '.3f') +
              ' 秒/update；峰值分配显存 ' + format(report['recent200_peak_allocated_bytes_max_rank']['max'] / 1024**3, '.2f') +
              ' GiB；辅助 query 梯度非零比例 ' + format(report['recent200_aux_query_nonzero_fraction'] * 100, '.2f') + '%。',
              '', '机制证据边界：', '']
    lines += ['- ' + line for line in report['mechanism_boundary']]
    lines += ['', '失败检查：`' + json.dumps(report['failed_checks']) + '`。这是 TRAIN 数值/合同诊断，不能当作 benchmark 效果。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-stem', type=Path, required=True)
    args = parser.parse_args()
    json_path = args.output_stem.with_suffix('.json')
    md_path = args.output_stem.with_suffix('.md')
    if json_path.exists() or md_path.exists():
        raise FileExistsError('New independent diagnostic paths required')
    report = diagnose()
    with json_path.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    with md_path.open('x') as stream:
        stream.write(render(report))
    print(json.dumps({'status': report['status'], 'common_cutoff': report['common_completed_update_cutoff'],
                      'failed_checks': report['failed_checks'], 'json': str(json_path), 'md': str(md_path)}))


if __name__ == '__main__':
    main()
