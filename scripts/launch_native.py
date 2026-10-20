#!/usr/bin/env python3
"""One finite reason-grounding run on RTX3090 GPUs0,1,3,5.

CPU plan default. A16-update benchmark never launches formal training. Signals
stop only this launcher's child process group; existing MiniLM is preserved.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
TRAIN_WORKER = Path(__file__).resolve().parents[1] / 'src/xai_integrity/train_native.py'
sys.path.insert(0, str(TRAIN_WORKER.parent))
import train_native as train

PROJECT = Path(os.environ.get('XAI_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
LEASE_ROOT = Path(os.environ.get('XAI_GPU_LEASE_ROOT', Path(__file__).resolve().parent / 'gpu_lease'))
LEASE_SHA = train.sha(LEASE_ROOT / 'eval_only_scheduler.py')
RESIDENT_SHA = train.sha(LEASE_ROOT / 'resident_policy_source.py')


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Missing MemAvailable')


def model_load_lock_path():
    return train.FORMAL / 'official_quality_20261009/.global_model_loader.lock'


def acquire_load_lock(state, statepath):
    handle = model_load_lock_path().open('a+')
    deadline = time.monotonic() + 900
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                memory = available_memory()
                if memory >= 200 * 1024**3:
                    state['load_lock_acquired_unix'] = time.time()
                    state['MemAvailable_before_load_bytes'] = memory
                    train.dump(statepath, state)
                    return handle
                fcntl.flock(handle, fcntl.LOCK_UN)
            except BlockingIOError:
                pass
            train.require(time.monotonic() < deadline, 'No safe CPU model-load slot within900s')
            time.sleep(1)
    except BaseException:
        handle.close()
        raise


def await_model_ready(child, stage, state, statepath):
    deadline = time.monotonic() + 900
    while True:
        train.require(child.poll() is None, 'Stage exited before four model-ready receipts')
        ready = []
        for rank in range(4):
            path = Path(stage['out']) / 'receipts' / ('RUN_rank%d.json' % rank)
            if path.exists():
                row = json.loads(path.read_text())
                if row.get('model_ready_unix') is not None:
                    train.require(row['original_model_parameters_sha256'] == train.ORIGINAL_PARAMETERS_SHA,
                                  'Model-ready initialization differs from original baseline')
                    ready.append({'rank': rank, 'model_ready_unix': row['model_ready_unix'],
                                  'original_model_parameters_sha256': row['original_model_parameters_sha256']})
        if len(ready) == 4:
            state.update(all_rank_model_ready=ready, load_lock_released_unix=time.time(), status='running')
            train.dump(statepath, state)
            return
        train.require(time.monotonic() < deadline, 'Model initialization exceeded900s')
        time.sleep(1)


def make_command(args, out):
    return [sys.executable, '-u', '-m', 'torch.distributed.run', '--nproc_per_node=4',
            '--nnodes=1', '--node_rank=0', '--master_addr=127.0.0.1',
            '--master_port=' + str(args.master_port),
            str(TRAIN_WORKER),
            '--mode', args.mode, '--benchmark-updates', str(args.benchmark_updates),
            '--coefficient', '.1', '--max-expanded-length', '1536',
            '--reason-module', str(args.reason_module.resolve()),
            '--reason-module-sha256', args.reason_module_sha256,
            '--out', str(out), '--execute-gpu']


def reap_owned_child(active, state, statepath):
    """Reap only our start_new_session child while the GPU lease is still held."""
    child = active[0]
    if child is None:
        return
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            code = child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = child.wait()
            state['owned_child_reap_escalated_to_SIGKILL'] = True
    else:
        code = child.wait()
    active[0] = None
    state['owned_child_reaped'] = {'pid': child.pid, 'returncode': code, 'reaped_unix': time.time()}
    train.dump(statepath, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['benchmark', 'main'], required=True)
    parser.add_argument('--benchmark-updates', type=int, default=16)
    parser.add_argument('--reason-module', type=Path, required=True)
    parser.add_argument('--reason-module-sha256', required=True)
    parser.add_argument('--gpus', default='0,1,3,5')
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--master-port', type=int, required=True)
    parser.add_argument('--execute-gpu', action='store_true')
    args = parser.parse_args()
    ids = [int(x) for x in args.gpus.split(',')]
    train.require(len(ids) == 4 and len(set(ids)) == 4, 'Select exactly four distinct GPUs')
    train.require(1024 <= args.master_port <= 65535, 'Invalid explicit master port')
    args.run_root = args.run_root.resolve()
    out = args.run_root / 'candidate_reason_grounding'
    check = argparse.Namespace(mode=args.mode, coefficient=.1, max_expanded_length=1536,
                               benchmark_updates=args.benchmark_updates,
                               reason_module=args.reason_module,
                               reason_module_sha256=args.reason_module_sha256, out=out)
    train.preflight(check)
    stage = {'coefficient': .1, 'out': str(out), 'command': make_command(args, out)}
    plan = {
        'schema': 'official_reason_grounding_launch_v1',
        'method_variant': 'official_reason_grounding', 'status': 'CPU_plan_only',
        'gpu_started': False, 'created_unix': time.time(), 'gpus': ids,
        'mode': args.mode, 'benchmark_updates': args.benchmark_updates,
        'seed': 42, 'coefficient': .1, 'stages': [stage],
        'train_worker_sha256': train.sha(TRAIN_WORKER),
        'source_snapshot_sha256': train.sha(Path(__file__).with_name('SOURCE_SNAPSHOT.json')),
        'reason_module_sha256': args.reason_module_sha256,
        'delta_policy': 'none forbenchmark; fixedfinal2109only formain; all27authorvalidations retained',
        'no_external_training_data': True, 'same_native_image_exposures': True,
        'additional_official_mask_supervision': True,
        'extra_LM_forwards': 0, 'extra_CLIP_forwards': 0,
        'extra_SAM_encoder_forwards': 0, 'additional_SAM_mask_queries': True,
        'auto_launch_main_after_benchmark': False, 'extra_compute_matched': False,
        'model_load_policy': 'shared lock until all4 model_ready; minimum200GiB MemAvailable before group load',
    }
    if not args.execute_gpu:
        print(json.dumps(plan, indent=2))
        return
    train.require(not args.run_root.exists(), 'Fresh output root required; no automatic retry/overwrite')
    filesystem = os.statvfs(train.FORMAL)
    available_bytes = filesystem.f_bavail * filesystem.f_frsize
    log_reserve_bytes = 256 * 1024**2
    # Original587427044 trainable bf16 elements plus serialization margin.
    expected_delta_upper_bytes = max(1_250_000_000, 587427044 * 2 + 128 * 1024**2)
    required_bytes = (train.RESERVE_BYTES + expected_delta_upper_bytes + log_reserve_bytes
                      if args.mode == 'main' else log_reserve_bytes)
    train.require(available_bytes >= required_bytes,
                  'Insufficient capacity for declared mode final delta/reserve/log budget')
    plan['disk_preflight'] = {'available_bytes': available_bytes, 'required_bytes': required_bytes,
                             'reserve_bytes': train.RESERVE_BYTES if args.mode == 'main' else 0,
                             'expected_delta_upper_bytes': expected_delta_upper_bytes if args.mode == 'main' else 0,
                             'log_reserve_bytes': log_reserve_bytes}
    train.require(train.sha(LEASE_ROOT / 'eval_only_scheduler.py') == LEASE_SHA and
                  train.sha(LEASE_ROOT / 'resident_policy_source.py') == RESIDENT_SHA,
                  'GPU lease/resident source changed')
    lease = train.imported(LEASE_ROOT / 'eval_only_scheduler.py', 'native_reason_gpu_lease')
    resident = train.imported(LEASE_ROOT / 'resident_policy_source.py', 'native_reason_resident')
    args.run_root.mkdir(parents=True)
    state = dict(plan, status='starting', gpu_started=True, launched=[])
    statepath = args.run_root / 'STATUS.json'
    train.dump(statepath, state)
    active = [None]
    def stop_own(signum, frame):
        # Cleanup occurs in the leased inner finally, before lock release.
        raise KeyboardInterrupt('Signal%d; interrupted own session only' % signum)
    signal.signal(signal.SIGTERM, stop_own)
    signal.signal(signal.SIGINT, stop_own)
    try:
        with lease.leased(ids, resident):
            try:
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, ids)), XAI_VISIBLE_GPUS=','.join(map(str, ids)), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           HF_HOME=str(PROJECT / 'assets/legion_hf_cache'),
                           TRANSFORMERS_CACHE=str(PROJECT / 'assets/legion_hf_cache'),
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           CUDA_HOME=os.environ.get('CUDA_HOME', '/usr/local/cuda'),
                           PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128', OMP_NUM_THREADS='4')
                cuda_home = Path(env['CUDA_HOME'])
                env['LD_LIBRARY_PATH'] = str(cuda_home / 'lib64') + ':' + env.get('LD_LIBRARY_PATH', '')
                env['PATH'] = os.environ.get('XAI_TRAIN_VENV', '') + ':' + os.environ.get('XAI_OFFICIAL_VENV', '') + ':' + str(cuda_home / 'bin') + ':' + env.get('PATH', '')
                ninja = subprocess.run(['ninja', '--version'], env=env, text=True, capture_output=True, check=True)
                state['ninja_preflight'] = {'returncode': ninja.returncode, 'stdout': ninja.stdout.strip(),
                                            'effective_PATH': env['PATH']}
                lease.check_cards(ids, resident)
                state.update(status='waiting_for_serial_model_load', load_lock_wait_started_unix=time.time(),
                             global_model_load_lock=str(model_load_lock_path()))
                train.dump(statepath, state)
                load_lock = acquire_load_lock(state, statepath)
                try:
                    lease.check_cards(ids, resident)
                    log = args.run_root / 'stage_0.log'
                    with log.open('xb') as stream:
                        child = subprocess.Popen(stage['command'], cwd=str(Path(__file__).parent), env=env,
                                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    active[0] = child
                    row = dict(stage, pid=child.pid, started_unix=time.time(), log=str(log))
                    state['launched'].append(row)
                    state['status'] = 'loading_models'
                    train.dump(statepath, state)
                    await_model_ready(child, stage, state, statepath)
                finally:
                    fcntl.flock(load_lock, fcntl.LOCK_UN)
                    load_lock.close()
                code = child.wait()
                row.update(returncode=code, ended_unix=time.time())
                active[0] = None
                train.require(code == 0, 'Training stage failed; no automatic retry')
                rank_receipts = []
                for rank in range(4):
                    path = out / 'receipts' / ('RUN_rank%d.json' % rank)
                    value = json.loads(path.read_text())
                    train.require(value['status'] == ('benchmark_complete' if args.mode == 'benchmark' else 'full_training_complete'),
                                  'Rank did not complete')
                    rank_receipts.append({
                        'rank': rank, 'path': str(path), 'sha256': train.sha(path),
                        'optimizer_updates': value['optimizer_updates'],
                        'reason_supervised_refs': value['reason_supervised_refs'],
                        'extra_SAM_mask_decode_calls': value['extra_SAM_mask_decode_calls'],
                        'extra_GT_pixel_exposures': value['extra_GT_pixel_exposures'],
                        'reason_auxiliary_gradient': value['reason_auxiliary_gradient'],
                        'visual_exposure_sha256': value['visual_exposure_sha256']})
                row['rank_receipts'] = rank_receipts
                state['status'] = 'benchmark_complete' if args.mode == 'benchmark' else 'full_complete'
            finally:
                reap_owned_child(active, state, statepath)
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        state['ended_unix'] = time.time()
        train.dump(statepath, state)


if __name__ == '__main__':
    main()
