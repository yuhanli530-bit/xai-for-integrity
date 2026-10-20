#!/usr/bin/env python3
"""Fresh native LEGION training with reason-span grounding, seed42/RTX3090 only.

One original LM forward/backward per microbatch. Extra mask queries reuse the
original SAM image embedding. CPU preflight is the default; no automatic main.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
FORMAL = Path(os.environ.get('XAI_FORMAL_ROOT', Path(__file__).resolve().parents[2]))
PREPARED = Path(os.environ.get('XAI_PREPARED_ROOT', FORMAL / 'prepared_full_s42_v2'))
SPEC_SHA = 'ed727a2d5b00b232d9cec1680a755722dc8b9aafafb71b6099f0ade4a13b65b3'
DELTA_SHA = 'a5c0504e1461f97a9a6dcf5387e8848f48119e66c632ced8866f1b5846d311f2'
ADAPTER_SHA = '57df6af8193af558340bd78cf7b5c00984ae0ad3f476206816ae14bd5d50a2da'
ORIGINAL_PARAMETERS_SHA = '89fba3d27b25d6942c8bbebff6f4f05a523d20b1fede27051e3bfaec8bc28f61'
RESERVE_BYTES = 8 * 1024**3
sys.path.insert(0, str(HERE / '_deps'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def jhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def dump(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value) + '\n')
        stream.flush()


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def preflight(args):
    require(args.mode in ('benchmark', 'main'), 'Explicit short check or main required')
    require(args.benchmark_updates == 16, 'Frozen16-update engineering check')
    require(args.coefficient == .1, 'One complete reason-grounding candidate; no baseline or coefficient sweep')
    require(args.max_expanded_length == 1536, 'Original author1536 context required')
    require(args.out.resolve() != PREPARED and PREPARED not in args.out.resolve().parents,
            'Never write into original baseline preparation')
    snapshot = json.loads((HERE / 'SOURCE_SNAPSHOT.json').read_text())
    require(bool(snapshot['files']), 'Frozen new method source snapshot required')
    for row in snapshot['files']:
        require(sha(FORMAL / row['path']) == row['sha256'], 'Frozen method dependency changed: ' + row['path'])
    require(sha(args.reason_module) == args.reason_module_sha256, 'Reason-grounding module changed')
    require(sha(PREPARED / 'TRAINING_SPEC.json') == SPEC_SHA, 'Original baseline spec changed')
    spec = json.loads((PREPARED / 'TRAINING_SPEC.json').read_text())
    fixed = {'mode': 'full', 'seed': 42, 'world_size': 4, 'micro_batch': 1,
             'gradient_accumulation': 4, 'effective_batch': 16, 'epochs': 3,
             'steps_per_epoch': 703, 'scheduled_optimizer_updates': 2109}
    require(all(spec[k] == v for k, v in fixed.items()), 'Original native setting changed')
    for row in spec['source_snapshot'] + spec['patched_sources']:
        require(sha(PREPARED / row['relative_path']) == row['sha256'], 'Original author source changed: ' + row['relative_path'])
    for split in (('train',) if args.mode == 'benchmark' else ('train', 'test')):
        row = spec['data_contract'][split]
        require(sha(row['annotation_path']) == row['annotation_sha256'], 'Original official annotation changed')
    require(sha(HERE / '_deps/delta_checkpoint.py') == DELTA_SHA, 'Verified delta helper changed')
    require(sha(HERE / '_deps/legion_text_adapter.py') == ADAPTER_SHA, 'Verified parameter adapter changed')
    return spec


def run(args, spec):
    import torch
    import torch.distributed as dist
    from delta_checkpoint import base_identity_from_prepared, capture_reference, save_delta
    from legion_text_adapter import assert_trainable_contract

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    require(world == 4 and torch.cuda.device_count() == 4, 'Exactly4 visible RTX3090 GPUs required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == os.environ.get('XAI_VISIBLE_GPUS', '0,1,3,5'), 'CUDA_VISIBLE_DEVICES must match XAI_VISIBLE_GPUS')
    require(all('3090' in torch.cuda.get_device_name(i) for i in range(world)), 'RTX3090 only')
    torch.cuda.set_device(local_rank)
    out = args.out.resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'receipts').mkdir()
    deadline = time.monotonic() + 90
    while not (out / 'receipts').exists():
        require(time.monotonic() < deadline, 'Rank0 failed to create fresh output')
        time.sleep(.05)
    receipts = out / 'receipts'
    os.environ['FORMAL_LEGION_RECEIPTS'] = str(receipts)
    sys.path[:0] = [str(PREPARED / 'source'), str(PREPARED)]
    train = imported(PREPARED / 'source/scripts/loc_exp/train.py', 'native_reason_grounding_author_train')
    argv = list(spec['author_arguments']) + ['--local_rank', str(local_rank)]
    if args.mode == 'benchmark':
        argv.append('--no_eval')
    cfg = train.parse_args(argv)
    cfg.formal_seed = 42
    cfg.formal_canary = False
    cfg.distributed = True
    cfg.log_base_dir = str(out / 'author_logs')
    cfg.exp_name = 'official_reason_grounding_s42'
    expected = {'batch_size': 1, 'grad_accumulation_steps': 4, 'epochs': 3,
                'steps_per_epoch': 703, 'epoch_samples': 11236, 'precision': 'bf16',
                'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': .05, 'lr': 1e-4,
                'beta1': .9, 'beta2': .95, 'ce_loss_weight': 1.,
                'dice_loss_weight': .2, 'bce_loss_weight': .4}
    require(all(getattr(cfg, k) == v for k, v in expected.items()), 'Original optimization changed')
    require(not cfg.resume and not cfg.auto_resume and not cfg.weight and
            cfg.version == spec['assets']['checkpoint'], 'Fresh original initialization required')
    updates = args.benchmark_updates if args.mode == 'benchmark' else 2109
    state = {
        'status': 'starting', 'mode': args.mode, 'rank': rank, 'world_size': 4,
        'seed': 42, 'effective_batch': 16, 'gradient_accumulation': 4,
        'scheduled_updates': 2109, 'native_exposure_budget': 33744,
        'updates_requested': updates, 'optimizer_updates': 0,
        'visual_forwards': 0, 'backward_calls': 0,
        'author_validation_calls': 0,
        'author_validation_calls_planned': 0 if args.mode == 'benchmark' else 27,
        'started_unix': time.time(), 'worker_sha256': sha(__file__),
        'source_snapshot_sha256': sha(HERE / 'SOURCE_SNAPSHOT.json'),
        'reason_module_sha256': args.reason_module_sha256,
        'original_spec_sha256': SPEC_SHA, 'extra_data': False,
        'additional_official_mask_supervision': True,
        'method_variant': 'official_reason_grounding', 'coefficient': .1,
        'max_expanded_length': 1536, 'reason_events': 0,
        'reason_supervised_refs': 0, 'extra_SAM_mask_decode_calls': 0,
        'extra_GT_pixel_exposures': 0, 'extra_LM_forwards': 0,
        'extra_CLIP_forwards': 0, 'extra_SAM_encoder_forwards': 0,
        'reason_auxiliary_gradient': {}, 'original_ds_config_unmodified': True,
        'checkpoint_policy': 'none_benchmark_or_fixed_final2109_model_delta_only',
        'optimizer_resume_supported': False,
        'comparison': 'Native data/init/modules/AdamW/batch/exposure matched; additional reason mask queries and mask GT exposure disclosed, not equal compute',
    }
    statepath = receipts / ('RUN_rank%d.json' % rank)
    dump(statepath, state)
    hooks = []
    try:
        train.seed_model(42)
        tokenizer = train.setup_tokenizer_and_special_tokens(cfg)
        model = train.initialize_model(cfg, tokenizer)
        train.prepare_model_for_training(model, tokenizer, cfg)
        state['trainable_contract_before'] = assert_trainable_contract(model, PREPARED / 'receipts/trainable_rank0.json')
        reason = imported(args.reason_module, 'official_native_reason_grounding_module')
        handle = reason.install_reason_grounding(
            model, tokenizer=tokenizer,
            annotation_path=spec['data_contract']['train']['annotation_path'],
            annotation_sha256=spec['data_contract']['train']['annotation_sha256'],
            source_dir=PREPARED / 'source', coefficient=.1,
            max_expanded_length=1536, strict_author=True)
        require(callable(getattr(handle, 'prepare', None)), 'Missing native reason preparation API')
        state['reason_install'] = {'installed': True, 'adds_parameters': False,
                                  'uses_native_forward': True, 'no_replay': True}
        state['trainable_contract_after'] = assert_trainable_contract(model, PREPARED / 'receipts/trainable_rank0.json')
        monitored = {
            'text_hidden_fcs': lambda name: 'text_hidden_fcs' in name,
            'SAM_hypernetwork': lambda name: 'mask_decoder.output_hypernetworks_mlps' in name,
            'LoRA_B': lambda name: 'lora_B' in name,
        }
        gradient = {key: {'calls': 0, 'finite': True, 'max_abs': 0., 'nonzero_calls': 0} for key in monitored}
        tracked = {key: [n for n, p in model.named_parameters() if p.requires_grad and predicate(n)]
                   for key, predicate in monitored.items()}
        require(all(tracked.values()), 'Missing original text projection/SAM hypernetwork/LoRA_B trainables')
        def gradient_hook(key):
            def record(value):
                stats = gradient[key]
                finite = bool(torch.isfinite(value).all())
                maximum = float(value.detach().float().abs().max())
                stats['calls'] += 1
                stats['finite'] &= finite
                stats['max_abs'] = max(stats['max_abs'], maximum)
                stats['nonzero_calls'] += int(maximum > 0)
                return value
            return record
        for key, names in tracked.items():
            for name, parameter in model.named_parameters():
                if name in names:
                    hooks.append(parameter.register_hook(gradient_hook(key)))
        engine, optimizer, scheduler = train.initialize_deepspeed(model, tokenizer, cfg)
        require(engine.zero_optimization_stage() == 2 and not engine.optimizer.cpu_offload,
                'Original ZeRO2/nooffload required')
        engine.train()
        initial_lora = {n: p.detach().cpu().clone() for n, p in engine.module.named_parameters() if 'lora_B' in n}
        if rank == 0:
            reference = capture_reference(engine, base_identity_from_prepared(PREPARED))
            dump(receipts / 'INITIAL_FULL_PARAMETER_REFERENCE.json', reference)
            digest = reference['original_model_parameters_sha256']
        else:
            reference, digest = None, None
        values = [digest]
        dist.broadcast_object_list(values, src=0)
        require(values[0] == ORIGINAL_PARAMETERS_SHA, 'Initial parameters differ from original baseline')
        state['original_model_parameters_sha256'] = values[0]
        state['model_ready_unix'] = time.time()
        dump(statepath, state)
        dist.barrier()
        train.seed_data(42)
        cap, reg, seg, validation = train.initialize_datasets_and_loaders(cfg, tokenizer)
        cap_loader, reg_loader, seg_loader, val_loader = train.setup_data_loaders(cfg, cap, reg, seg, validation, tokenizer)
        require(cap_loader is None and reg_loader is None and len(seg) == 11236,
                'Original official TRAIN loader only')
        require((val_loader is None) == (args.mode == 'benchmark'), 'Validation mode mismatch')
        iterator = iter(seg_loader)
        writer = train.initialize_environment(cfg)
        epoch_seeds = [random.randint(0, 100000) for _ in range(cfg.epochs)]
        from tools.utils import dict_to_cuda
        completed, micro_count = 0, 0
        visual_digest = hashlib.sha256()
        def validate(epoch, trigger):
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                result = train.validate_model_performance(val_loader, engine, epoch, writer, cfg)
            finally:
                engine.train()
            torch.cuda.synchronize()
            state['author_validation_calls'] += 1
            append(receipts / ('AUTHOR_VALIDATION_rank%d.jsonl' % rank), {
                'call': state['author_validation_calls'], 'epoch': epoch + 1,
                'optimizer_updates': completed, 'trigger': trigger,
                'seconds': time.perf_counter() - started,
                'original_function': True, 'reason_disabled_under_no_grad': True,
                'rng_restored_or_reseeded': False, 'result': result})
            dump(statepath, state)
        for epoch in range(cfg.epochs):
            engine.train()
            random.seed(epoch_seeds[epoch])
            random.choices([0], weights=[cfg.weight_segm], k=cfg.steps_per_epoch)
            for step in range(cfg.steps_per_epoch):
                if completed >= updates:
                    break
                torch.cuda.synchronize()
                started_update = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                for micro in range(cfg.grad_accumulation_steps):
                    started_micro = time.perf_counter()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(seg_loader)
                        batch = next(iterator)
                    train.audit_batch(batch, cfg, epoch, step, micro)
                    exposure = {'micro': micro_count, 'image_paths': batch['image_paths'],
                                'input_ids': batch['input_ids'].tolist(), 'labels': batch['labels'].tolist(),
                                'conversation_sha256': [hashlib.sha256(v.encode()).hexdigest() for v in batch['conversation_list']],
                                'mask_sha256': [hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in batch['masks_list']]}
                    exposure_hash = jhash(exposure)
                    visual_digest.update(exposure_hash.encode())
                    append(receipts / ('VISUAL_EXPOSURE_rank%d.jsonl' % rank), {
                        'micro': micro_count, 'exposure_sha256': exposure_hash,
                        'image_paths': batch['image_paths'],
                        'native_conversation_sha256': exposure['conversation_sha256']})
                    batch = dict_to_cuda(batch)
                    for key in ('global_enc_images', 'grounding_enc_images'):
                        if batch[key] is not None:
                            batch[key] = batch[key].bfloat16()
                    preparation = handle.prepare(batch, completed + 1, micro)
                    require(isinstance(preparation, dict), 'Preparation must return JSON-safe receipt')
                    torch.cuda.synchronize()
                    started_forward = time.perf_counter()
                    visual = engine(**batch)
                    require(bool(torch.isfinite(visual['loss'])), 'Nonfinite total loss')
                    torch.cuda.synchronize()
                    forward_seconds = time.perf_counter() - started_forward
                    receipt = handle.last_receipt
                    for field in ('extra_LM_forwards', 'extra_CLIP_forwards', 'extra_SAM_encoder_forwards'):
                        require(int(receipt.get(field, 0)) == 0, 'Unexpected extra image/LM encoder call: ' + field)
                    losses = {k: float(v.detach()) for k, v in visual.items()
                              if hasattr(v, 'detach') and v.numel() == 1}
                    loss_for_audit = visual['loss'].detach()
                    started_backward = time.perf_counter()
                    engine.backward(visual['loss'])
                    torch.cuda.synchronize()
                    backward_seconds = time.perf_counter() - started_backward
                    require(not any(engine.optimizer.params_already_reduced), 'Native backward reduction did not complete')
                    auxiliary_gradient = dict(handle.auxiliary_gradient)
                    require(auxiliary_gradient.get('finite', True), 'Nonfinite auxiliary pooled-query gradient')
                    require(all(row['finite'] for row in gradient.values()), 'Nonfinite native module gradient')
                    supervised_refs = int(receipt.get('additional_GT_mask_exposures', 0))
                    state['reason_events'] += int(supervised_refs > 0)
                    state['reason_supervised_refs'] += supervised_refs
                    state['extra_SAM_mask_decode_calls'] += int(receipt.get('extra_decoder_forwards', 0))
                    state['extra_GT_pixel_exposures'] += int(receipt.get('additional_GT_pixel_exposures', 0))
                    state['reason_auxiliary_gradient'] = auxiliary_gradient
                    append(receipts / ('REASON_rank%d.jsonl' % rank), {
                        'update': completed + 1, 'micro': micro_count,
                        'micro_in_update': micro, 'preparation': preparation,
                        'receipt': receipt, 'auxiliary_gradient': auxiliary_gradient,
                        'total_module_gradient': gradient, 'losses': losses})
                    del visual
                    started_step = time.perf_counter()
                    engine.step()
                    torch.cuda.synchronize()
                    step_seconds = time.perf_counter() - started_step
                    train.audit_step(loss_for_audit, engine, cfg, epoch, step, micro)
                    append(receipts / ('TIMING_rank%d.jsonl' % rank), {
                        'update': completed + 1, 'micro': micro_count,
                        'native_plus_reason_forward_seconds': forward_seconds,
                        'native_plus_reason_backward_seconds': backward_seconds,
                        'step_seconds': step_seconds,
                        'whole_micro_seconds': time.perf_counter() - started_micro,
                        'warmup': completed < 1, 'supervised_refs': supervised_refs,
                        'extra_SAM_mask_decode_calls': int(receipt.get('extra_decoder_forwards', 0)),
                        'extra_GT_pixel_exposures': int(receipt.get('additional_GT_pixel_exposures', 0)),
                        'extra_LM_forwards': 0, 'extra_CLIP_forwards': 0,
                        'extra_SAM_encoder_forwards': 0, **losses})
                    state['visual_forwards'] += 1
                    state['backward_calls'] += 1
                    micro_count += 1
                    del batch, loss_for_audit
                completed += 1
                require(int(engine.global_steps) == completed and not engine.skipped_steps,
                        'Skipped optimizer update')
                norm = engine.get_global_grad_norm()
                norm_value = float(norm) if norm is not None else None
                require(norm_value is not None and math.isfinite(norm_value), 'Missing/nonfinite gradient norm')
                state['last_gradient_norm'] = norm_value
                timing = torch.tensor([time.perf_counter() - started_update, torch.cuda.max_memory_allocated()],
                                      dtype=torch.float64, device=engine.device)
                dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                append(receipts / ('UPDATES_rank%d.jsonl' % rank), {
                    'update': completed, 'warmup': completed <= 1,
                    'seconds_max_rank': timing[0].item(),
                    'peak_allocated_bytes_max_rank': timing[1].item(),
                    'learning_rates': scheduler.get_last_lr(),
                    'total_module_gradient': gradient,
                    'reason_auxiliary_gradient': auxiliary_gradient,
                    'global_gradient_norm': norm_value})
                state.update(status='running', optimizer_updates=completed,
                             visual_exposure_sha256=visual_digest.hexdigest(),
                             total_module_gradient=gradient)
                dump(statepath, state)
                if args.mode == 'main' and step % 100 == 0:
                    validate(epoch, 'author_step_mod100_zero')
            if args.mode == 'main' and step + 1 == cfg.steps_per_epoch:
                train.finish_epoch(cfg, epoch)
                validate(epoch, 'author_epoch_end')
                torch.cuda.synchronize()
                dist.barrier()
                if rank == 0:
                    if completed != 2109:
                        append(receipts / 'CHECKPOINT_EVENTS.jsonl', {
                            'epoch': epoch + 1, 'global_steps': completed,
                            'status': 'save_skipped_for_storage',
                            'fixed_final_selection_unmodified': True})
                    else:
                        started_save = time.perf_counter()
                        with (FORMAL / '.final_delta_save.lock').open('a+') as save_lock:
                            fcntl.flock(save_lock, fcntl.LOCK_EX)
                            manifest = save_delta(engine, out / 'checkpoint/global_step2109', reference,
                                metadata={'epoch': 3, 'global_steps': 2109,
                                          'method_variant': 'official_reason_grounding',
                                          'coefficient': .1, 'max_expanded_length': 1536,
                                          'reason_module_sha256': args.reason_module_sha256,
                                          'run_status': state}, reserve_bytes=RESERVE_BYTES)
                        append(receipts / 'CHECKPOINT_EVENTS.jsonl', {
                            'epoch': 3, 'global_steps': 2109, 'status': 'final_delta_saved',
                            'seconds': time.perf_counter() - started_save,
                            'weights_bytes': manifest['weights_bytes'],
                            'weights_sha256': manifest['weights_sha256'],
                            'omitted_optimizer_resume_state': True})
                dist.barrier()
            if completed >= updates:
                break
        changed_lora = [n for n, p in engine.module.named_parameters()
                        if n in initial_lora and not torch.equal(initial_lora[n], p.detach().cpu())]
        require(completed == updates and bool(changed_lora), 'No genuine optimizer updates')
        require(state['reason_supervised_refs'] > 0, 'No actual reason mask supervision')
        require(handle.auxiliary_gradient.get('finite', False) and
                handle.auxiliary_gradient.get('nonzero_calls', 0) > 0,
                'No actual finite nonzero auxiliary pooled-query gradient')
        require(all(row['finite'] and row['nonzero_calls'] > 0 for row in gradient.values()),
                'No actual finite nonzero text projection/SAM hypernetwork/LoRA_B gradient')
        if args.mode == 'main':
            require(state['author_validation_calls'] == 27, 'Missing original author validations')
        state.update(status='benchmark_complete' if args.mode == 'benchmark' else 'full_training_complete',
                     changed_lora_B=changed_lora, total_module_gradient=gradient,
                     reason_auxiliary_gradient=dict(handle.auxiliary_gradient))
        if writer is not None:
            writer.close()
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        for hook in hooks:
            hook.remove()
        state['ended_unix'] = time.time()
        dump(statepath, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['benchmark', 'main'], required=True)
    parser.add_argument('--benchmark-updates', type=int, default=16)
    parser.add_argument('--coefficient', type=float, default=.1)
    parser.add_argument('--max-expanded-length', type=int, default=1536)
    parser.add_argument('--reason-module', type=Path, required=True)
    parser.add_argument('--reason-module-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute-gpu', action='store_true')
    args = parser.parse_args()
    spec = preflight(args)
    if not args.execute_gpu:
        print(json.dumps({'status': 'CPU_preflight_pass', 'gpu_started': False,
                         'mode': args.mode, 'seed': 42, 'world_size': 4,
                         'effective_batch': 16, 'training_updates': args.benchmark_updates if args.mode == 'benchmark' else 2109,
                         'scheduler_updates': 2109, 'coefficient': .1,
                         'author_validation_calls': 0 if args.mode == 'benchmark' else 27,
                         'checkpoint_policy': 'none' if args.mode == 'benchmark' else 'final2109_only',
                         'worker_sha256': sha(__file__), 'reason_module_sha256': args.reason_module_sha256,
                         'method_variant': 'official_reason_grounding',
                         'extra_LM_forwards': 0, 'extra_CLIP_forwards': 0,
                         'extra_SAM_encoder_forwards': 0,
                         'additional_SAM_mask_queries': True,
                         'additional_GT_mask_exposure': True,
                         'extra_compute_matched': False}, indent=2))
        return
    run(args, spec)


if __name__ == '__main__':
    main()
