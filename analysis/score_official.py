#!/usr/bin/env python3
"""Coverage/provenance wrapper only. Calls the unchanged frozen baseline scorer."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from common import PROJECT, TEST_DIR, ORDER_SHA, sha, read, new, require, binding, digest

TEST_SHA = '3bb55380f9ad75d76c26ad9a18035296be4ab2a7255b5e2fa11b131a2417e7e4'
PINNED = {
 'formal_legion_20261007/c_final_eval_v1/score_official_with_provenance.py':'4a2c439368fc03169884516520743634182ebc809d00c5aa3e5bd0147325d436',
 'official_legion_20261006/metrics.py':'5bab86f477922166f661624589f6d5dce1b8109d0650624e89a2656b9ddf8337',
 'official_legion_20261006/SUPPORT_ASSETS.json':'40e03385266f783c3054c91ee9a363ccaf52f337d7e29b7ad4da17e94ecf851b',
 'repos/LEGION/eval/utils.py':'496093ad23b6050c5006109ec1ef6a1ba31db00de0a9e9325338cd04dfd6e0fe'}


def inspect_shards(directories, bound):
    loaded = [(Path(p).resolve(), read(Path(p)/'RUN_RECEIPT.json')) for p in directories]
    require(bool(loaded), 'No completed shards')
    ref = loaded[0][1]; names = ref['full_image_order']; n = ref['num_shards']
    require(len(names) == len(set(names)) == 1000 and digest(names) == ORDER_SHA and names == sorted(names), 'Official1000 list differs')
    require(len(loaded) == n and sorted(r['shard_index'] for _,r in loaded) == list(range(n)), 'Every unique shard index required')
    require(ref['full_image_order_sha256'] == digest(names) and ref['full_image_content_sha256'] == digest(ref['full_images']) and
            [r['name'] for r in ref['full_images']] == names, 'Image content manifest differs')
    seen = set(); artifacts = []
    for directory,row in loaded:
        plan = read(directory/'PLAN.json')
        require(all(row.get(k)==v for k,v in plan.items()), 'Shard plan/receipt disagreement')
        require(row['binding'] == bound and row['status']=='complete' and row['raw_failed']==0 and
                row.get('parameters_unchanged') is True and row.get('sources_unchanged') is True,
                'Failed/incomplete/changed-model shard cannot receive a full score')
        for k in ('full_image_order','full_images','full_image_order_sha256','full_image_content_sha256','num_shards','equivalence_receipt_sha256'):
            require(row[k] == ref[k], 'Shards used different model/data/witness')
        expected = names[row['shard_index']::n]
        require(row['shard_image_order']==expected and row['raw_completed_images']==expected and
                row['raw_completed']==len(expected) and row['planned']==len(expected) and
                row['raw_metrics_denominator']==len(expected), 'Missing or duplicated raw denominator')
        native, memory = row['native_completed_images'], row['memory_recovered_images']
        require(len(native)==row['native_completed'] and len(memory)==row['memory_recovered'] and
                len(set(native+memory))==len(expected) and set(native+memory)==set(expected) and not(set(native)&set(memory)) and
                set(memory)==set(row['native_oom_images']) and not row['recovery_failed_images'], 'Invalid actual-OOM recovery history')
        require(sorted(p.name for p in (directory/'predictions').iterdir())==sorted(expected), 'Unexpected/missing predictions')
        for name in expected:
            require(name not in seen, 'Duplicate image across shards'); seen.add(name)
            folder = directory/'predictions'/name
            require((folder/'mask.png').is_file(), 'Missing union mask')
            text = read(folder/'explanation.json')
            require(list(text)==[str(TEST_DIR/name)] and isinstance(next(iter(text.values())),str), 'Original explanation format changed')
            artifacts.append(dict(image=name, union_mask_sha256=sha(folder/'mask.png'), explanation_sha256=sha(folder/'explanation.json')))
    require(seen==set(names), 'Not exactly full1000')
    return loaded, artifacts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run-dir',type=Path,required=True); p.add_argument('--training-status',type=Path,required=True)
    p.add_argument('--shard-dir',type=Path,action='append',required=True); p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(); require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Scoring is CPU-only')
    bound=binding(args.training_run_dir,args.training_status)
    loaded,artifacts=inspect_shards(args.shard_dir,bound)
    annotation=PROJECT/'data/synthscars/SynthScars/test/annotations/test.json'
    require(sha(annotation)==TEST_SHA, 'Official test annotation changed')
    for path,wanted in PINNED.items(): require(sha(PROJECT/path)==wanted,'Frozen scoring source changed: '+path)
    require(importlib.metadata.version('rouge-score')=='0.1.2','Original ROUGE version required')
    require(not args.output_dir.exists(),'New score output required'); args.output_dir.mkdir(parents=True)
    complete=bound['completion']; weights=Path(complete['delta'])/'model_delta.pt'
    method='official-native-reason-grounding-seed42-final2109'
    shards=[]
    for index,(directory,row) in enumerate(loaded):
        receipt=args.output_dir/('SCORING_COMPATIBILITY_%d.json'%index)
        new(receipt,dict(status='complete',method_id=method,planned=row['planned'],completed=row['raw_completed'],failed=0,
                         model_state_sha256=complete['delta_weights_sha256'],source_receipt_sha256=sha(directory/'RUN_RECEIPT.json')))
        shards.append(dict(run=str(directory),receipt=str(receipt),planned=row['planned']))
    manifest=dict(method_id=method,seed=42,planned=1000,checkpoint=dict(model_state=str(weights.resolve()),
                  model_state_sha256=complete['delta_weights_sha256'],global_steps=2109,kind='all297trainables_and_buffers_delta'),
                  shards=shards,official_protocol=dict(test_annotation=str(annotation),test_annotation_sha256=TEST_SHA),
                  generation=bound['generation'],reason_coefficient=complete['coefficient'],
                  reason_module_sha256=complete['reason_module_sha256'],
                  method_variant=complete['method_variant'],
                  auxiliary_scope='official referent grounding, not explanation validity',
                  extra_training_compute_matched=False,
                  training_completion=complete,baseline_scoring_sources_sha256=PINNED,
                  checkpoint_selection='fixed final epoch3/2109; reason-grounding declared after earlier method test results; no within-run test checkpoint selection',
                  raw_output_policy='original union of all masks>0 and unchanged explanation; no quality gating',
                  artifacts=artifacts)
    plan=args.output_dir/'OFFICIAL_SCORE_PLAN.json'; new(plan,manifest)
    result=args.output_dir/'OFFICIAL_SCORE.json'
    command=[sys.executable,str(PROJECT/'formal_legion_20261007/c_final_eval_v1/score_official_with_provenance.py'),
             '--project-root',str(PROJECT),'--runs',*[str(d) for d,_ in loaded],'--out',str(result),'--method-id',method,
             '--checkpoint-path',str(weights),'--checkpoint-sha256',complete['delta_weights_sha256'],'--evaluation-manifest',str(plan)]
    new(args.output_dir/'SCORER_COMMAND.json',command); subprocess.run(command,check=True)
    report=read(result); require(report.get('status')=='complete' and report.get('completed')==1000,'Scorer incomplete')
    new(args.output_dir/'SCORING_RECEIPT.json',dict(status='complete',score_sha256=sha(result),denominator=1000,
        source_sha256=PINNED,actual_rouge_version=importlib.metadata.version('rouge-score'),calibration_claim=False,
        metrics=['miou_two_classes','foreground_f1','rouge_l_f1','css']))
    print(json.dumps(dict(status='complete',score=str(result))))


if __name__=='__main__':main()
