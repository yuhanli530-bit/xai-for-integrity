#!/usr/bin/env python3
"""One finite post-training witness -> full1000 shards -> frozen official score.

No training, retries, cron, method selection, or unrelated-process signals.
Before the exact four-rank final completion gate passes, no TEST files are read.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import gpu_idle

from common import HERE, PROJECT, sha, read, new, update, require, imported, terminal_training, source_contract, GPU_GROUPS

LEASE_ROOT=HERE/'gpu_lease'
LEASE_SHA=sha(LEASE_ROOT/'eval_only_scheduler.py')
RESIDENT_SHA=sha(LEASE_ROOT/'resident_policy_source.py')


def environment(gpus):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),CUDA_DEVICE_ORDER='PCI_BUS_ID',
               HF_HOME=str(PROJECT/'assets/legion_hf_cache'),TRANSFORMERS_CACHE=str(PROJECT/'assets/legion_hf_cache'),
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_HOME=os.environ.get('CUDA_HOME','/usr/local/cuda'),
               PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128',OMP_NUM_THREADS='4')
    cuda_home=Path(env['CUDA_HOME'])
    env['LD_LIBRARY_PATH']=str(cuda_home/'lib64')+':'+env.get('LD_LIBRARY_PATH','')
    env['PATH']=os.environ.get('XAI_TRAIN_VENV','')+':'+os.environ.get('XAI_OFFICIAL_VENV','')+':'+str(cuda_home/'bin')+':'+env.get('PATH','')
    return env


def verify_snapshot(expected):
    require(sha(HERE/'SNAPSHOT.json')==expected,'Evaluation snapshot manifest changed')
    for row in read(HERE/'SNAPSHOT.json')['files']:
        path=Path(row['path']);require(not path.is_absolute() and '..' not in path.parts,'Unsafe snapshot path')
        require(sha(HERE/path)==row['sha256'] and (HERE/path).stat().st_size==row['bytes'],'Evaluation source changed: '+str(path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',choices=('G',),required=True)
    p.add_argument('--training-run-dir',type=Path,required=True);p.add_argument('--training-status',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--gpus',required=True)
    p.add_argument('--snapshot-sha256',required=True)
    p.add_argument('--wait-seconds',type=int,default=0)
    p.add_argument('--evaluation-timeout-seconds',type=int,default=86400)
    p.add_argument('--execute-gpu',action='store_true')
    args=p.parse_args();ids=[int(x) for x in args.gpus.split(',')]
    require(ids==GPU_GROUPS[args.candidate],'Use the fixed reason-grounding GPU group')
    require(args.training_run_dir.name=='candidate_reason_grounding','Candidate/run directory differs')
    require(0<=args.wait_seconds<=86400 and 1<=args.evaluation_timeout_seconds<=172800,'Finite deadlines required')
    verify_snapshot(args.snapshot_sha256)
    require(sha(LEASE_ROOT/'eval_only_scheduler.py')==LEASE_SHA and sha(LEASE_ROOT/'resident_policy_source.py')==RESIDENT_SHA,
            'Frozen shared GPU lease/resident policy changed')
    source_contract()  # source hashes only, never reads test annotations/images
    completed=terminal_training(args.training_run_dir,args.training_status)
    if args.training_status.is_file():require(read(args.training_status).get('method_variant')=='official_reason_grounding','Method/status differs')
    plan=dict(status='ready_for_witness' if completed else 'pending_training_completion',gpu_started=False,
              training_run_dir=str(args.training_run_dir.resolve()),training_status=str(args.training_status.resolve()),
              output_dir=str(args.output_dir.resolve()),gpus=ids,candidate=args.candidate,seed=42,full_test_denominator=1000,
              native_first=True,memory_recovery='one retry after actual OOM only after exact same-delta witness',
              new_training_jobs=0,automatic_retries=0,cron=False,test_files_read=False,
              snapshot_sha256=args.snapshot_sha256)
    if not args.execute_gpu: print(json.dumps(plan,indent=2));return
    require(args.training_status.is_file(),'Explicit existing training launcher status required')
    require(not args.output_dir.exists(),'New queue output required; no restart/overwrite')
    args.output_dir.mkdir(parents=True);new(args.output_dir/'PLAN.json',plan)
    state=dict(plan,started_unix=time.time(),stages=[]);active=[];stopped=[False]
    def stop_children():
        for child in active:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
    def stop_own(signum,frame):stopped[0]=True;stop_children()
    signal.signal(signal.SIGTERM,stop_own);signal.signal(signal.SIGINT,stop_own)
    status=args.output_dir/'STATUS.json'
    def reap_owned():
        for child in active:
            if child.poll() is None:
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait()
    def idle_event(value):
        with (args.output_dir/'GPU_IDLE_SNAPSHOTS.jsonl').open('a') as stream:
            stream.write(json.dumps(value)+'\n'); stream.flush()
    def stage(name,command,gpus):
        require(not stopped[0],'Controller stopped before launch')
        verify_snapshot(args.snapshot_sha256)
        logfile=args.output_dir/(name+'.log')
        with logfile.open('xb') as stream:
            child=subprocess.Popen(command,cwd=str(HERE),env=environment(gpus),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        active.append(child)
        # Signal can occur while Popen returns: register first, then honor it.
        if stopped[0]:stop_children();raise InterruptedError('Interrupted around child creation')
        row=dict(name=name,pid=child.pid,command=command,gpus=gpus,started_unix=time.time(),log=str(logfile))
        state['stages'].append(row);state['status']='running_'+name;update(status,state)
        return child,row
    def finish(jobs,deadline):
        remaining=list(jobs)
        while remaining:
            require(not stopped[0],'Queue interrupted')
            require(time.monotonic()<deadline,'Finite evaluation timeout')
            for child,row in list(remaining):
                code=child.poll()
                if code is None:continue
                row.update(returncode=code,ended_unix=time.time());remaining.remove((child,row));update(status,state)
                require(code==0,'Evaluation stage failed; preserve logs; no automatic retry: '+row['name'])
            if remaining:time.sleep(2)
    try:
        deadline=time.monotonic()+args.wait_seconds
        while completed is None:
            require(not stopped[0],'Queue interrupted while awaiting training')
            require(time.monotonic()<deadline,'Training completion pending at finite deadline')
            state['status']='waiting_exact_training_completion';update(status,state);time.sleep(min(15,max(0,deadline-time.monotonic())))
            completed=terminal_training(args.training_run_dir,args.training_status)
        new(args.output_dir/'TRAINING_COMPLETION.json',completed)
        deadline=time.monotonic()+args.evaluation_timeout_seconds
        # Import ONLY the stable lease utilities. Its old evaluation main is never invoked.
        lease=imported(LEASE_ROOT/'eval_only_scheduler.py','quality_eval_gpu_lease')
        lease._refresh_uuids()
        resident=imported(LEASE_ROOT/'resident_policy_source.py','quality_eval_resident_policy')
        with gpu_idle.leased(ids,lease.GPU_UUIDS,resident,idle_event,lambda:stopped[0]):
            try:
                state['gpu_lease']={'indices':ids,'checked_snapshots':str(args.output_dir/'GPU_IDLE_SNAPSHOTS.jsonl')};state['gpu_started']=True;update(status,state)
                prefix=[sys.executable,'-u',str(HERE/'native_eval.py'),'--training-run-dir',str(args.training_run_dir),
                        '--training-status',str(args.training_status),'--execute-gpu']
                witness=args.output_dir/'witness'
                finish([stage('witness',prefix+['--mode','witness','--output-dir',str(witness)],[ids[0]])],deadline)
                gpu_idle.wait_idle(ids,lease.GPU_UUIDS,resident,idle_event,lambda:stopped[0])
                jobs=[];dirs=[]
                for index,gpu in enumerate(ids):
                    dest=args.output_dir/('shard_%d'%index);dirs.append(dest)
                    cmd=prefix+['--mode','shard','--num-shards',str(len(ids)),'--shard-index',str(index),
                                '--equivalence-receipt',str(witness/'EQUIVALENCE_RECEIPT.json'),'--output-dir',str(dest)]
                    jobs.append(stage('shard_%d'%index,cmd,[gpu]))
                finish(jobs,deadline)
            except BaseException:
                stop_children();reap_owned();raise  # Keep GPU leases until all owned children exit.
        cmd=[sys.executable,'-u',str(HERE/'score.py'),'--training-run-dir',str(args.training_run_dir),
             '--training-status',str(args.training_status),'--output-dir',str(args.output_dir/'score')]
        for dest in dirs:cmd.extend(['--shard-dir',str(dest)])
        finish([stage('official_score',cmd,[])],deadline)
        state.update(status='full1000_official_score_complete',score=str(args.output_dir/'score/OFFICIAL_SCORE.json'))
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc));stop_children();raise
    finally:
        # Reap only our children. Training/service processes are never in active.
        reap_owned()
        state['ended_unix']=time.time();update(status,state)
    print(json.dumps(dict(status=state['status'],score=state['score'])))


if __name__=='__main__':main()
