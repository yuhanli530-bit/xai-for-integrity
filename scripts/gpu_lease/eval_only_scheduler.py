"""Cooperative GPU guard: check occupancy and serialize local launches.

Never signals, kills, or reconfigures processes. The lock coordinates only
users of this repository; use a cluster scheduler for multi-user allocation.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess

LOCK = Path(os.environ.get("XAI_GPU_LOCK", "/tmp/xai-for-integrity-gpu.lock"))


def _gpu_uuids(ids):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    pairs = {}
    for line in raw.splitlines():
        index, uuid = [part.strip() for part in line.split(",", 1)]
        pairs[int(index)] = uuid
    return {pairs[i] for i in ids}


def check_cards(ids, resident=None):
    wanted = _gpu_uuids(ids)
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    active = [line.strip() for line in raw.splitlines() if line.strip() and line.split(",", 1)[0].strip() in wanted]
    if active:
        raise RuntimeError("Requested GPUs have active compute processes; use the cluster scheduler or wait: " + "; ".join(active))


@contextmanager
def leased(ids, resident=None):
    held = []
    try:
        for index in sorted(ids):
            held.append(resident.acquire_gpu_lock(index))
        check_cards(ids, resident)
        yield
    finally:
        for fd in reversed(held):
            os.close(fd)
