#!/usr/bin/env python3
"""Create deterministic referent/control occlusion images from a manifest CSV.

Input columns: image_id,source_group,claim_id,image_path,mask_path. The binary
mask identifies the claimed referent. Outputs are fresh and never overwrite.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--control-attempts", type=int, default=300)
    args = p.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Output directory must be fresh")
    args.output_dir.mkdir(parents=True)
    rows_out = []
    with args.manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        image_path, mask_path = Path(row["image_path"]), Path(row["mask_path"])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.NEAREST)
        m = np.asarray(mask) >= 128
        if not m.any():
            rows_out.append({**row, "status": "failed", "failure": "empty_referent_mask"})
            continue
        array = np.asarray(image).copy()
        fill = tuple(int(v) for v in np.median(array[~m], axis=0)) if (~m).any() else (127, 127, 127)
        variants = {"referent_occlusion": m.copy()}
        coords = np.argwhere(~m)
        area = int(m.sum())
        side = max(1, int(area ** .5))
        seed = int(hashlib.sha256(row["claim_id"].encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        control = None
        h, w = m.shape
        for _ in range(args.control_attempts):
            y = rng.randrange(max(1, h - side + 1)); x = rng.randrange(max(1, w - side + 1))
            candidate = np.zeros_like(m); candidate[y:min(h, y + side), x:min(w, x + side)] = True
            overlap = int((candidate & m).sum())
            if overlap <= max(1, int(.02 * area)) and candidate.sum() > 0:
                control = candidate
                break
        if control is None:
            rows_out.append({**row, "status": "failed", "failure": "no_matched_control_region_found"})
            continue
        variants["matched_control_occlusion"] = control
        for name, region in variants.items():
            changed = array.copy()
            changed[region] = fill
            out = args.output_dir / row["source_group"] / row["claim_id"] / f"{name}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(changed).save(out)
            rows_out.append({**row, "status": "complete", "intervention": name,
                             "input_image_sha256": sha(image_path), "mask_sha256": sha(mask_path),
                             "output_path": str(out), "output_sha256": sha(out),
                             "occluded_pixels": int(region.sum()), "fill_rgb": list(fill)})
    receipt = args.output_dir / "manifest.jsonl"
    with receipt.open("x") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"input_rows": len(rows), "output_rows": len(rows_out),
                      "complete": sum(r.get("status") == "complete" for r in rows_out),
                      "failed": sum(r.get("status") != "complete" for r in rows_out),
                      "manifest": str(receipt)}, indent=2))


if __name__ == "__main__":
    main()
