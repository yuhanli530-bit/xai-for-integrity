#!/usr/bin/env python3
"""Freeze evaluator/analysis source bytes into scripts/SNAPSHOT.json."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INCLUDE = [
    Path("common.py"), Path("gpu_idle.py"), Path("native_eval.py"),
    Path("native_helpers.py"), Path("score.py"), Path("evaluate_official.py"),
    Path("SOURCE_SNAPSHOT.json"), Path("_deps/delta_checkpoint.py"),
    Path("_deps/legion_text_adapter.py"), Path("gpu_lease/eval_only_scheduler.py"),
    Path("gpu_lease/resident_policy_source.py"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = [{"path": str(p), "bytes": (ROOT / p).stat().st_size, "sha256": digest(ROOT / p)} for p in INCLUDE]
    target = ROOT / "SNAPSHOT.json"
    target.write_text(json.dumps({"schema": "xai-for-integrity-evaluator-snapshot-v1", "files": rows}, indent=2) + "\n")
    print(digest(target))


if __name__ == "__main__":
    main()
