"""Fixed native LEGION evaluation contracts; no image/model reads on import."""
import hashlib
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('XAI_PROJECT_ROOT', Path(__file__).resolve().parents[2]))
PREPARED = PROJECT / 'formal_legion_20261007/prepared_full_s42_v2'
SPEC_SHA = 'ed727a2d5b00b232d9cec1680a755722dc8b9aafafb71b6099f0ade4a13b65b3'
TEST_DIR = PROJECT / 'data/synthscars/SynthScars/test/images'
ORDER_SHA = '674636c88a12e0fc419e0dd9fce7bd5eb2efc3b3109a77e4f968428cb093522e'
TRAIN_IMAGE = PROJECT / 'data/synthscars/SynthScars/train/images/b5fe117f-f051-4491-a817-c8dcdd27ae2b.jpg'
TRAIN_SHA = 'a7e6b5c1e4aebf4412fa6eee2fe4442628b339fc64accfa2c76e7dc318b24dbc'
MEMORY = PROJECT / 'official_legion_20261006/memory_eval.py'
MEMORY_SHA = '0791fac90ac5aa373f3687b8e4210a85428352ed54cfa213e26c6999cbb10f76'
EXPECTED_INITIAL_SHA = '89fba3d27b25d6942c8bbebff6f4f05a523d20b1fede27051e3bfaec8bc28f61'
EXPECTED_WORKER_SHA = 'd711e3f3bf3b15e4fcc09320a5d64e52828d628a90961a69a35711f99ebf2ebc'
EXPECTED_OBJECTIVE_SHA = '6bfa73ac681b7c9527153264ffb44044238e183570e12e7d7e937a404361f799'
EXPECTED_TRAIN_SNAPSHOT_SHA = 'c18e801c5605f9d76bb2d3409d93f1f246c820476d7d2a4d82a1db31f0359d70'
TRAIN_ANNOTATION_SHA = '94100c0b381f4b24745d75e5cba65872686ccf9e37af8e9fd669349e3509cd27'
GPU_GROUPS = {'G': [0,1,3,5]}
TRAIN_SNAPSHOT = PROJECT/'formal_legion_20261007/official_reason_grounding_20261010/runner/revision1/SOURCE_SNAPSHOT.json'
DEPS = {'delta_checkpoint.py': 'a5c0504e1461f97a9a6dcf5387e8848f48119e66c632ced8866f1b5846d311f2',
        'legion_text_adapter.py': '281293c11d99487f8934248cc60d30867036792b7ea5b43d6b26975e3e93757d'}
PROMPT = 'Please provide a detailed analysis of artifacts in this photo, considering physical artifacts (e.g., optical display issues, violations of physical laws, and spatial/perspective errors), structural artifacts (e.g., deformed objects, asymmetry, or distorted text), and distortion artifacts (e.g., color/texture distortion, noise/blur, artistic style errors, and material misrepresentation). Output with interleaved segmentation masks for the corresponding parts of the answer.'
GENERATION = dict(prompt=PROMPT, max_new_tokens=512, num_beams=1, do_sample=False,
                  use_cache=False, mask_threshold=0, sam_longest_side=1024, batch_size=1)
EXACT = ('token_ids_equal', 'raw_text_equal', 'text_equal', 'phrases_equal',
         'seg_count_equal', 'mask_shape_equal', 'binary_masks_equal', 'masks_bitwise_equal', 'finite_masks')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def new(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, indent=2, ensure_ascii=False); stream.write('\n')


def update(path, value):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


@contextlib.contextmanager
def model_load_lease(path):
    """Cross-process CPU load lease; unrelated to GPU inference duration."""
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def audit_reason_log(path, row):
    completed = refs = pixels = calls = entries = 0
    for index, line in enumerate(Path(path).open()):
        event = json.loads(line); receipt = event['receipt']; entries += 1
        require(event['micro'] == index and event['update'] == index // 4 + 1 and
                event['micro_in_update'] == index % 4, 'Native micro/update order differs')
        require(receipt['api'] == 'prepare_native_forward_reason_grounding_one_extra_decoder_v1' and
                receipt['method_variant'] == 'official_reason_grounding' and receipt['coefficient'] == .1,
                'Reason objective event differs')
        require(all(receipt.get(key) == 0 for key in ('extra_LM_forwards','extra_CLIP_forwards',
                    'extra_SAM_encoder_forwards','extra_backward_calls','extra_projection_forwards')),
                'Unexpected additional encoder/replay computation')
        if receipt.get('completed'):
            n = receipt['additional_GT_mask_exposures']
            require(receipt.get('native_exact_join') is True and
                    receipt.get('official_annotation_sha256') == TRAIN_ANNOTATION_SHA and
                    receipt.get('current_target_token_hidden_read') is False and
                    receipt.get('auxiliary_query_requires_grad') is True and
                    n > 0 and n == len(receipt['selected_refs']) == receipt['auxiliary_mask_count'],
                    'Missing exact explanation/ref/GT binding')
            require(receipt['extra_decoder_forwards'] == receipt['extra_prompt_encoder_forwards'] == 1 and
                    receipt['additional_GT_pixel_exposures'] > 0,
                    'Actual mask query / pixel exposures missing')
            for ref in receipt['selected_refs']:
                require(ref['native_explanation_fully_supervised'] is True and
                        ref['native_ref_SEG_position'] < ref['native_explanation_token_begin'] <
                        ref['native_explanation_token_end'] <= receipt['native_retained_tokens'],
                        'Explanation predictive span binding differs')
            grad = receipt.get('auxiliary_query_gradient', {})
            require(grad.get('finite') is True and type(grad.get('nonzero')) is bool,
                    'Missing finite actual auxiliary backward receipt')
            completed += 1; refs += n; calls += 1; pixels += receipt['additional_GT_pixel_exposures']
        else:
            require(receipt['extra_decoder_forwards'] == receipt['additional_GT_mask_exposures'] == 0,
                    'Skipped auxiliary performed unreported work')
    require(entries == 8436 and completed == row['reason_events'] and refs == row['reason_supervised_refs'] and
            calls == row['extra_SAM_mask_decode_calls'] and pixels == row['extra_GT_pixel_exposures'],
            'Reason logs and completed counters disagree')
    return dict(path=str(path), sha256=sha(path), entries=entries, completed=completed,
                supervised_refs=refs, additional_GT_pixel_exposures=pixels)


def terminal_training(run_dir, launcher_status, *, verify_weights=True):
    run_dir, launcher_status = Path(run_dir).resolve(), Path(launcher_status).resolve()
    if not launcher_status.is_file(): return None
    launch = read(launcher_status)
    require(launch.get('mode') == 'main' and launch.get('schema') == 'official_reason_grounding_launch_v1' and
            launch.get('method_variant') == 'official_reason_grounding' and launch.get('coefficient') == .1 and
            launch.get('gpus') == GPU_GROUPS['G'], 'Wrong native reason-grounding launcher')
    require(launch.get('status') != 'failed', 'Training failed; do not evaluate')
    if launch.get('status') != 'full_complete': return None
    stages = launch.get('launched', [])
    require(len(stages) == 1 and Path(stages[0]['out']).resolve() == run_dir and
            stages[0].get('returncode') == 0, 'Launcher did not finish exact training run')
    fixed = dict(status='full_training_complete',mode='main',world_size=4,seed=42,effective_batch=16,
                 gradient_accumulation=4,scheduled_updates=2109,native_exposure_budget=33744,
                 updates_requested=2109,optimizer_updates=2109,visual_forwards=8436,backward_calls=8436,
                 author_validation_calls=27,extra_data=False,additional_official_mask_supervision=True,
                 original_ds_config_unmodified=True,method_variant='official_reason_grounding',
                 coefficient=.1,max_expanded_length=1536,extra_LM_forwards=0,extra_CLIP_forwards=0,
                 extra_SAM_encoder_forwards=0)
    rows = []
    for rank in range(4):
        path = run_dir/'receipts'/('RUN_rank%d.json'%rank); row = read(path)
        require(row.get('rank') == rank and all(row.get(k) == v for k,v in fixed.items()),
                'Final native setting differs on rank%d'%rank)
        require(row.get('original_spec_sha256') == SPEC_SHA and
                row.get('original_model_parameters_sha256') == EXPECTED_INITIAL_SHA and
                row.get('worker_sha256') == EXPECTED_WORKER_SHA and
                row.get('reason_module_sha256') == EXPECTED_OBJECTIVE_SHA and
                row.get('source_snapshot_sha256') == EXPECTED_TRAIN_SNAPSHOT_SHA,
                'Source or original initialization differs')
        before, after = row['trainable_contract_before'], row['trainable_contract_after']
        require(before == after and before['trainable_tensors'] == 297 and
                before['trainable_elements'] == 587427044 and row.get('changed_lora_B'), 'Trainables/updates differ')
        grad = row['reason_auxiliary_gradient']
        require(grad.get('finite') is True and grad.get('nonzero_calls',0) > 0 and
                row['reason_supervised_refs'] > 0, 'No real auxiliary training evidence')
        require(all(g['finite'] and g['nonzero_calls'] > 0 for g in row['total_module_gradient'].values()),
                'Missing native text/LM/SAM gradients')
        audit = audit_reason_log(run_dir/'receipts'/('REASON_rank%d.jsonl'%rank), row)
        keys = ('reason_events','reason_supervised_refs','extra_SAM_mask_decode_calls','extra_GT_pixel_exposures')
        rows.append(dict(rank=rank,sha256=sha(path),reason_log=audit,**{key:row[key] for key in keys}))
    require(launch['train_worker_sha256'] == EXPECTED_WORKER_SHA and
            launch['reason_module_sha256'] == EXPECTED_OBJECTIVE_SHA and
            launch['source_snapshot_sha256'] == EXPECTED_TRAIN_SNAPSHOT_SHA,
            'Launcher/rank source differs')
    delta = run_dir/'checkpoint/global_step2109'; manifest = read(delta/'MANIFEST.json')
    meta, saved = manifest['metadata'], manifest['metadata']['run_status']
    require(manifest['format'] == 'legion-all-trainable-model-delta-v1' and
            manifest['weights_file'] == 'model_delta.pt' and meta['epoch'] == 3 and meta['global_steps'] == 2109 and
            meta['method_variant'] == 'official_reason_grounding' and meta['coefficient'] == .1 and
            meta['reason_module_sha256'] == EXPECTED_OBJECTIVE_SHA, 'Wrong fixed-final method delta')
    require(saved['optimizer_updates'] == 2109 and saved['author_validation_calls'] == 27 and
            saved['original_model_parameters_sha256'] == EXPECTED_INITIAL_SHA and
            saved['source_snapshot_sha256'] == EXPECTED_TRAIN_SNAPSHOT_SHA and
            all(saved[key] == rows[0][key] for key in keys), 'Saved final metadata differs')
    weights = delta/'model_delta.pt'
    require(weights.stat().st_size == manifest['weights_bytes'], 'Delta size changed')
    if verify_weights: require(sha(weights) == manifest['weights_sha256'], 'Delta bytes changed')
    return dict(status='full_training_and_final_delta_verified',training_run_dir=str(run_dir),
                launcher_status=str(launcher_status),launcher_status_sha256=sha(launcher_status),rank_receipts=rows,
                method='official_native_reason_grounding',method_variant='official_reason_grounding',coefficient=.1,
                reason_module_sha256=EXPECTED_OBJECTIVE_SHA,extra_compute_matched=False,
                original_spec_sha256=SPEC_SHA,delta=str(delta),delta_manifest_sha256=sha(delta/'MANIFEST.json'),
                delta_weights_sha256=manifest['weights_sha256'],epoch=3,global_steps=2109)


def source_contract():
    require(sha(TRAIN_SNAPSHOT) == EXPECTED_TRAIN_SNAPSHOT_SHA, 'Frozen training snapshot changed')
    for row in read(TRAIN_SNAPSHOT)['files']:
        require(sha(PROJECT/'formal_legion_20261007'/row['path']) == row['sha256'], 'Frozen reason-grounding training source changed')
    require(sha(PREPARED/'TRAINING_SPEC.json') == SPEC_SHA, 'Original baseline spec changed')
    spec = read(PREPARED/'TRAINING_SPEC.json')
    for row in spec['source_snapshot'] + spec['patched_sources']:
        require(sha(PREPARED/row['relative_path']) == row['sha256'], 'Frozen author source changed')
    for name, expected in DEPS.items(): require(sha(HERE/'_deps'/name) == expected, 'Delta reconstruction helper changed')
    require(sha(MEMORY) == MEMORY_SHA, 'Validated memory evaluator changed')
    files = [*HERE.glob('*.py'), *(HERE/'_deps').glob('*.py')]
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(files)}


def binding(run_dir, launcher_status):
    complete = terminal_training(run_dir, launcher_status)
    require(complete is not None, 'Training has not completed; no test read permitted')
    return dict(completion=complete, sources=source_contract(), generation=GENERATION,
                memory_eval_sha256=MEMORY_SHA, witness_train_sha256=TRAIN_SHA)


def test_images(bound):
    require(bound['completion']['status'] == 'full_training_and_final_delta_verified', 'Training gate first')
    paths = sorted(p for p in TEST_DIR.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
    names = [p.name for p in paths]
    require(len(names) == len(set(names)) == 1000 and digest(names) == ORDER_SHA, 'Official full1000 identity changed')
    return paths, [dict(name=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in paths]


def require_witness(path, bound):
    row = read(path)
    require(row.get('status') == 'pass' and row.get('binding') == bound and
            row.get('sources_unchanged') is True and row.get('parameters_unchanged') is True and
            all(row.get('comparison', {}).get(k) is True for k in EXACT), 'Exact same-delta native/memory witness required')
    return sha(path)
