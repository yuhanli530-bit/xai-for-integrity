#!/usr/bin/env python3
"""Build intact/referent/control triplets with equal-area, same-shape masks.

CSV columns: image_id,source_group,claim_id,claim_text,image_path,mask_path.
Paths are relative to the manifest directory. Masks must be binary and aligned;
no implicit resizing. Unplaceable controls remain recorded failures.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

CONDITIONS = ('intact', 'referent_occlusion', 'matched_control_occlusion')
KEY_FIELDS = ('source_group', 'image_id', 'claim_id')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def matched_control(mask, key, attempts):
    """Translate the entire mask without overlap or clipping; never reshape it."""
    if attempts < 1:
        raise ValueError('control-attempts must be positive')
    ys, xs = np.where(mask)
    if not len(ys):
        raise ValueError('empty_referent_mask')
    y0, x0 = int(ys.min()), int(xs.min())
    patch = mask[y0:int(ys.max())+1, x0:int(xs.max())+1]
    h, w = mask.shape
    ph, pw = patch.shape
    ny, nx = h-ph+1, w-pw+1
    rng = random.Random(int(hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16], 16))
    for position in rng.sample(range(ny*nx), min(attempts, ny*nx)):
        y, x = divmod(position, nx)
        if np.any(mask[y:y+ph, x:x+pw] & patch):
            continue
        control = np.zeros_like(mask)
        control[y:y+ph, x:x+pw] = patch
        return control, {'dx': x-x0, 'dy': y-y0}
    raise ValueError('disjoint_same_shape_control_search_exhausted')


def build(manifest, output_dir, attempts=300):
    manifest, output_dir = Path(manifest), Path(output_dir)
    if attempts < 1:
        raise ValueError('control-attempts must be positive')
    with manifest.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = (*KEY_FIELDS, 'claim_text', 'image_path', 'mask_path')
        if set(required)-set(reader.fieldnames or []):
            raise ValueError(f'Required manifest columns: {required}')
        rows = list(reader)
    if not rows:
        raise ValueError('Manifest must not be empty')
    seen = set()
    for row in rows:
        if any(not (row[k] or '').strip() for k in required):
            raise ValueError('Manifest contains empty required fields')
        key = tuple(row[k] for k in KEY_FIELDS)
        if key in seen:
            raise ValueError(f'Duplicate planned claim: {key}')
        seen.add(key)
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_rows, validity_rows = [], []
    complete_claims = 0
    for index, row in enumerate(rows):
        key = tuple(row[k] for k in KEY_FIELDS)
        validity_rows.append({**{k: row[k] for k in (*KEY_FIELDS, 'claim_text')},
                              'validity_status': 'pending', 'validity_annotator_id': '',
                              'intact_supported': '', 'referent_supported': '', 'control_supported': ''})
        try:
            image_path = (manifest.parent / row['image_path']).resolve()
            mask_path = (manifest.parent / row['mask_path']).resolve()
            with Image.open(image_path) as original:
                array = np.array(original.convert('RGB'))
            with Image.open(mask_path) as original:
                mask_array = np.array(original.convert('L'))
            if mask_array.shape != array.shape[:2]:
                raise ValueError('mask_image_size_mismatch')
            values = set(np.unique(mask_array).tolist())
            if not (values <= {0, 1} or values <= {0, 255}):
                raise ValueError('mask_must_be_binary_0_1_or_0_255')
            mask = mask_array > 0
            control, translation = matched_control(mask, key, attempts)
            fill = np.median(array[~mask], axis=0).astype(np.uint8)
            variants = {'intact': np.zeros_like(mask), 'referent_occlusion': mask,
                        'matched_control_occlusion': control}
            claim_dir = output_dir / f'{index:06d}'
            claim_dir.mkdir()
            pending = []
            for condition, region in variants.items():
                changed = array.copy()
                changed[region] = fill
                path = claim_dir / f'{condition}.png'
                region_path = claim_dir / f'{condition}_mask.png'
                Image.fromarray(changed).save(path)
                Image.fromarray(region.astype(np.uint8)*255).save(region_path)
                pending.append({**row, 'status': 'complete', 'intervention': condition,
                                'input_image_sha256': sha(image_path), 'mask_sha256': sha(mask_path),
                                'output_path': str(path.resolve()), 'output_sha256': sha(path),
                                'intervention_mask_path': str(region_path.resolve()),
                                'occluded_pixels': int(region.sum()),
                                'changed_pixels': int(np.any(changed != array, axis=2).sum()),
                                'referent_overlap_pixels': int((region & mask).sum()),
                                'fill_rgb': fill.tolist(), 'control_translation': translation})
            receipt_rows.extend(pending)
            complete_claims += 1
        except (ValueError, OSError) as error:
            receipt_rows.extend({**row, 'status': 'failed', 'intervention': c, 'failure': str(error)}
                                for c in CONDITIONS)
    with (output_dir / 'manifest.jsonl').open('x') as stream:
        for row in receipt_rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    with (output_dir / 'validity_template.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(validity_rows[0]))
        writer.writeheader()
        writer.writerows(validity_rows)
    summary = {'schema': 'intervention-generation-v2', 'status': 'complete' if complete_claims == len(rows) else 'incomplete_denominator',
               'planned_claims': len(rows), 'planned_rows': 3*len(rows),
               'complete_claims': complete_claims, 'failed_claims': len(rows)-complete_claims,
               'control': 'translated identical binary mask; exact area; zero referent overlap',
               'control_attempts': attempts,
               'validity': 'pending human review; geometrical matching does not imply semantic validity'}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--control-attempts', type=int, default=300)
    args = parser.parse_args()
    try:
        result = build(args.manifest, args.output_dir, args.control_attempts)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    if result['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
