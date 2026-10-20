"""Bounded utilization settling; physical identity/process/memory checks stay strict."""
import contextlib
import csv
import os
import subprocess
import time


def snapshot():
    def query(fields, kind):
        raw = subprocess.check_output(['nvidia-smi', '--query-' + kind + '=' + fields,
                                       '--format=csv,noheader,nounits'], text=True)
        return [[x.strip() for x in row] for row in csv.reader(raw.splitlines()) if row]
    return {'observed_unix': time.time(),
            'gpus': query('index,uuid,name,memory.used,utilization.gpu', 'gpu'),
            'apps': query('pid,gpu_uuid,used_memory', 'compute-apps')}


def inspect(value, indices, uuids, resident):
    observed = {int(r[0]): r for r in value['gpus']}
    selected, busy = [], []
    for index in indices:
        r = observed[index]
        if r[1] != uuids[index] or '3090' not in r[2]:
            raise RuntimeError('Physical GPU identity/type differs: ' + str(index))
        apps = [x for x in value['apps'] if x[1] == r[1]]
        if index == 0 and apps:
            if len(apps) != 1 or apps[0][0] != '853283' or int(r[3]) >= 500:
                raise RuntimeError('GPU0 has unapproved process/memory: ' + str(apps))
            resident.resident_policy(resident.proc_identity(853283))
        elif apps or int(r[3]) >= 100:
            raise RuntimeError('GPU has process or memory allocation: ' + str(index))
        utilization = int(r[4])
        if not 0 <= utilization <= 100:
            raise RuntimeError('Invalid utilization sample: ' + str(index))
        if utilization: busy.append(index)
        selected.append({'index': index, 'uuid': r[1], 'memory_mib': int(r[3]),
                         'utilization_percent': utilization, 'compute_apps': apps})
    return {'gpus': selected, 'utilization_nonzero_indices': busy}


def wait_idle(indices, uuids, resident, emit, stopped=lambda: False, timeout=60,
              poll_seconds=2, read=snapshot, clock=time.monotonic, sleep=time.sleep):
    """Require two idle samples; only utilization may settle within the finite window."""
    deadline = clock() + timeout
    consecutive = 0
    while True:
        if stopped(): raise InterruptedError('Stopped while waiting for idle GPUs')
        value = read()
        try:
            checked = inspect(value, indices, uuids, resident)
        except BaseException as exc:
            emit(dict(value, status='hard_guard_failure', error_type=type(exc).__name__, error=str(exc)))
            raise
        consecutive = consecutive + 1 if not checked['utilization_nonzero_indices'] else 0
        emit(dict(value, status='idle' if not checked['utilization_nonzero_indices'] else 'settling_utilization',
                  selected=checked, consecutive_idle_samples=consecutive))
        if consecutive >= 2: return checked
        if clock() >= deadline: raise RuntimeError('GPU utilization did not settle within finite idle deadline')
        sleep(min(poll_seconds, max(0, deadline - clock())))


@contextlib.contextmanager
def leased(indices, uuids, resident, emit, stopped):
    held = []
    try:
        for index in sorted(indices): held.append(resident.acquire_gpu_lock(index))
        wait_idle(indices, uuids, resident, emit, stopped)
        yield
    finally:
        for fd in reversed(held): os.close(fd)
