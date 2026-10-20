#!/usr/bin/env python3
"""Native same-delta TRAIN witness or fixed full1000 shard. GPU flag is explicit."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

from common import (HERE, PROJECT, PREPARED, MEMORY, TRAIN_IMAGE, TRAIN_SHA, PROMPT, EXACT,
                    sha, read, new, require, imported, binding, digest, test_images, require_witness)
from common import model_load_lease
from native_helpers import capture_native, compare_outputs, runtime_config


def attempt(model, tokenizer, author_call):
    import torch
    try: return capture_native(model, tokenizer, author_call), None
    except Exception as exc:
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (isinstance(exc, RuntimeError) and 'CUDA out of memory' in str(exc))
        return None, dict(error_type=type(exc).__name__, error=str(exc), actual_cuda_oom=oom)


def make_plan(args):
    bound = binding(args.training_run_dir, args.training_status)
    plan = dict(schema='official_native_reason_grounding_eval_v1', mode=args.mode, binding=bound,
                ground_truth_inputs=False, quality_gating=False, extra_model_pass=False)
    if args.mode == 'witness':
        require(sha(TRAIN_IMAGE) == TRAIN_SHA, 'Fixed training witness image changed')
        plan.update(source_image=str(TRAIN_IMAGE), source_image_sha256=TRAIN_SHA,
                    test_images_read=False, test_annotations_read=False, exact_requirements=list(EXACT))
        return plan
    require(1 <= args.num_shards <= 8 and 0 <= args.shard_index < args.num_shards, 'Invalid fixed shard partition')
    require(args.equivalence_receipt is not None, 'Witness receipt path required')
    witness_sha = require_witness(args.equivalence_receipt, bound) if args.equivalence_receipt.is_file() else None
    if args.execute_gpu: require(witness_sha is not None, 'Cannot execute before exact same-delta witness PASS')
    images, records = test_images(bound); names = [p.name for p in images]
    plan.update(full_image_order=names, full_images=records, full_image_order_sha256=digest(names),
                full_image_content_sha256=digest(records), full_denominator=1000,
                num_shards=args.num_shards, shard_index=args.shard_index,
                shard_image_order=names[args.shard_index::args.num_shards],
                equivalence_receipt_sha256=witness_sha,
                native_first=True, memory_recovery='one retry only after actual native CUDA OOM')
    return plan


def execute(args, plan):
    require(not args.output_dir.exists(), 'New output directory required; no resume or overwrite')
    args.output_dir.mkdir(parents=True); new(args.output_dir/'PLAN.json', plan)
    sys.path.insert(0, str(HERE/'_deps'))
    import random
    import numpy as np
    import cv2
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor
    from legion_text_adapter import load_original_author_model
    from delta_checkpoint import load_delta, base_identity_from_prepared
    model = None; original_evaluate = None
    receipt = dict(status='failed', started_unix=time.time(), **plan)
    completed, failed, native, recovered, oom_images, recovery_failed, rows = [], [], [], [], [], [], []
    try:
        require(torch.cuda.device_count() == 1 and '3090' in torch.cuda.get_device_name(0), 'One visible RTX3090 required')
        # Both four-GPU queues share this lock, independent of snapshot/output paths.
        # Only transient CPU-heavy reconstruction is serialized; inference is not.
        lock_path = PROJECT/'formal_legion_20261007/official_quality_20261009/.global_model_loader.lock'
        loading=dict(lock=str(lock_path),waiting_unix=time.time())
        new(args.output_dir/'LOAD_WAIT.json',loading)
        with model_load_lease(lock_path):
            loading['acquired_unix']=time.time()
            torch.cuda.set_device(0)
            model, tokenizer, cfg, base = load_original_author_model(PREPARED, args.output_dir/'load_receipts', local_rank=0)
            restored = load_delta(model, Path(plan['binding']['completion']['delta']),
                                 expected_base_identity=base_identity_from_prepared(PREPARED), align_runtime_dtypes=True)
            model = model.cuda().eval()
            gc.collect();torch.cuda.synchronize()
            loading.update(status='model_cuda_ready_CPU_temporaries_released',ready_unix=time.time())
            new(args.output_dir/'LOAD_READY.json',loading)
        receipt['runtime'] = runtime_config(model)
        if args.mode == 'shard': require(receipt['runtime'] == read(args.equivalence_receipt)['runtime'], 'Witness runtime differs')
        new(args.output_dir/'MODEL_RESTORED.json', dict(base=base, restored=restored, runtime=receipt['runtime']))
        original_evaluate = model.__class__.evaluate
        versions = {n:p._version for n,p in model.named_parameters()}
        inference = imported(PREPARED/'source/scripts/loc_exp/infer.py', 'official_quality_native_author_infer')
        memory = imported(MEMORY, 'official_quality_validated_memory_infer')
        from model.SAM.utils.transforms import ResizeLongestSide
        processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower, local_files_only=True)
        transform = ResizeLongestSide(1024)
        def call_for(path):
            def call(current):
                image = cv2.imread(str(path))
                require(image is not None, 'Cannot decode image: '+str(path))
                return inference.inference(current, PROMPT, image.astype(np.uint8), tokenizer, processor, transform, cfg)
            return call
        def rng(): return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        def restore(value):
            random.setstate(value[0]); np.random.set_state(value[1]); torch.set_rng_state(value[2]); torch.cuda.set_rng_state_all(value[3])
        if args.mode == 'witness':
            outputs, timing = {}, {}; state = rng()
            for mode in ('native', 'memory'):
                model.__class__.evaluate = original_evaluate
                if mode == 'memory': memory.install()
                restore(state); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                start = time.monotonic()
                outputs[mode] = capture_native(model, tokenizer, call_for(TRAIN_IMAGE))
                torch.cuda.synchronize()
                timing[mode] = dict(seconds=time.monotonic()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated())
                new(args.output_dir/(mode+'_output.json'), {k:v for k,v in outputs[mode].items() if k != 'masks'})
                np.savez_compressed(args.output_dir/(mode+'_masks.npz'), masks=outputs[mode]['masks'].float().numpy())
            receipt.update(comparison=compare_outputs(outputs['native'], outputs['memory'], model.seg_token_idx), measurements=timing)
        else:
            from common import TEST_DIR
            (args.output_dir/'predictions').mkdir(); (args.output_dir/'raw').mkdir()
            expected = {r['name']:r for r in plan['full_images']}
            for name in plan['shard_image_order']:
                path = TEST_DIR/name
                require(sha(path) == expected[name]['sha256'], 'Test bytes changed after manifest freeze')
                model.__class__.evaluate = original_evaluate
                start = time.monotonic(); torch.cuda.reset_peak_memory_stats(); state = rng()
                output, error = attempt(model, tokenizer, call_for(path)); initial_error = error; mode = 'native'
                if error is not None and error['actual_cuda_oom']:
                    oom_images.append(name); gc.collect(); torch.cuda.empty_cache(); memory.install(); restore(state)
                    output, error = attempt(model, tokenizer, call_for(path)); mode = 'memory_after_native_cuda_oom'
                    model.__class__.evaluate = original_evaluate
                    if error is not None: recovery_failed.append(name)
                if error is None:
                    dest = args.output_dir/'predictions'/name; dest.mkdir()
                    binary = (output['masks'] > 0)
                    Image.fromarray(torch.any(binary,dim=0).numpy().astype('uint8')*255, mode='L').save(dest/'mask.png')
                    new(dest/'explanation.json', {str(path):output['text']})
                    rawdest = args.output_dir/'raw'/name; rawdest.mkdir()
                    new(rawdest/'text.json', {k:v for k,v in output.items() if k != 'masks'})
                    # Phrase masks remain ordered; mismatch is recorded, never guessed.
                    phrase_rows = []
                    aligned = len(output['phrases']) == len(binary)
                    for index,mask in enumerate(binary):
                        filename = 'mask_%03d.png'%index
                        Image.fromarray(mask.numpy().astype('uint8')*255, mode='L').save(rawdest/filename)
                        phrase_rows.append(dict(mask_index=index, file=filename, phrase=output['phrases'][index] if aligned else None))
                    new(rawdest/'phrase_masks.json', dict(count_aligned=aligned, masks=phrase_rows))
                    completed.append(name); (native if mode=='native' else recovered).append(name)
                    del output, binary
                else: failed.append(dict(image=name, **error))
                row = dict(image=name, status='complete' if error is None else 'failed', raw_mode=mode,
                           native_error=initial_error, final_error=error, seconds=time.monotonic()-start,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(), denominator_contribution=1)
                rows.append(row)
                with (args.output_dir/'PREDICTIONS.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n'); stream.flush()
                gc.collect(); torch.cuda.empty_cache()
        receipt['parameters_unchanged'] = versions == {n:p._version for n,p in model.named_parameters()}
        receipt['sources_unchanged'] = binding(args.training_run_dir, args.training_status) == plan['binding']
        require(receipt['parameters_unchanged'] and receipt['sources_unchanged'], 'Evaluation changed model/source binding')
        if args.mode == 'witness': receipt['status'] = 'pass' if all(receipt['comparison'][k] for k in EXACT) else 'fail'
        else: receipt['status'] = 'complete' if not failed else 'raw_incomplete'
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        if model is not None and original_evaluate is not None: model.__class__.evaluate = original_evaluate
        receipt['ended_unix'] = time.time()
        if args.mode == 'shard': receipt.update(planned=len(plan['shard_image_order']), raw_completed=len(completed),
            raw_completed_images=completed, raw_failed=len(failed), raw_failures=failed, native_completed=len(native),
            native_completed_images=native, memory_recovered=len(recovered), memory_recovered_images=recovered,
            native_oom_images=oom_images, recovery_failed_images=recovery_failed, raw_metrics_denominator=len(plan['shard_image_order']),
            unattempted_images=[name for name in plan['shard_image_order'] if name not in {r['image'] for r in rows}])
        new(args.output_dir/('EQUIVALENCE_RECEIPT.json' if args.mode=='witness' else 'RUN_RECEIPT.json'), receipt)
        if model is not None: del model
        gc.collect(); torch.cuda.empty_cache()
    print(json.dumps(dict(status=receipt['status'], output_dir=str(args.output_dir))))
    require(receipt['status'] in ('pass','complete'), 'Evaluation did not pass; outputs/failures retained')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['witness','shard'], required=True)
    ap.add_argument('--training-run-dir', type=Path, required=True); ap.add_argument('--training-status', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--equivalence-receipt', type=Path)
    ap.add_argument('--num-shards', type=int, default=1); ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--execute-gpu', action='store_true')
    args = ap.parse_args(); plan = make_plan(args)
    if not args.execute_gpu: print(json.dumps(dict(status='CPU_preflight_only', gpu_started=False, **plan), indent=2)); return
    execute(args, plan)


if __name__ == '__main__': main()
