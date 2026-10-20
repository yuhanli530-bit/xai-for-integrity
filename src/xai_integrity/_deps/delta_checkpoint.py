"""Exact model-only delta checkpoints for native LEGION, not optimizer resume.

The original base is required. All trainable parameters and ALL registered buffers
(including nonpersistent RoPE caches) are stored without quantization. Frozen
parameter hashes prove that they are still reconstructible from that base.
Call on rank zero only, after an optimizer boundary with all ranks synchronized.
No weights, CUDA contexts, environments, or files are changed by importing this file.
"""
from __future__ import annotations
import hashlib
import inspect
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import uuid

FORMAT = 'legion-all-trainable-model-delta-v1'
EXPECTED_TENSORS = 297
EXPECTED_ELEMENTS = 587427044


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def _core(model):
    """Unwrap only known wrappers; stage 3 requires a separate collective gather."""
    if type(model).__name__ == 'DeepSpeedEngine':
        if int(model.zero_optimization_stage()) > 2:
            raise ValueError('ZeRO3 is unsupported: its parameters may be partitioned')
        model = model.module
    while type(model).__name__ in ('DistributedDataParallel', 'DataParallel', 'PeftModelForCausalLM', 'PeftModel'):
        model = model.get_base_model() if hasattr(model, 'get_base_model') else model.module
    for parameter in model.parameters():
        if hasattr(parameter, 'ds_id'):
            raise ValueError('Partitioned ZeRO3 parameter detected')
    return model


def _rank_zero():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        raise RuntimeError('Only global rank zero writes model delta; synchronize outside this function')


def _tensor_copy(tensor):
    # clone is essential: a ZeRO2 parameter may view a much larger flat storage.
    return tensor.detach().to(device='cpu').contiguous().clone()


def tensor_sha256(tensor):
    import torch
    value = tensor.detach().to(device='cpu').contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    flat = value.numpy()
    for start in range(0, len(flat), 8 * 1024**2):
        digest.update(memoryview(flat[start:start + 8 * 1024**2]))
    return digest.hexdigest()


def _tensor_row(tensor, with_hash=True):
    row = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
           'elements': tensor.numel(), 'bytes': tensor.numel() * tensor.element_size()}
    if with_hash:
        row['sha256'] = tensor_sha256(tensor)
    return row


def _parameters(model):
    return dict(model.named_parameters())


def _module_paths(model, prefix='', ancestors=()):
    # torch1.13 named_parameters lacks remove_duplicate; inspect registrations.
    if id(model) in ancestors:
        raise ValueError('Cyclic module registration is unsupported')
    yield prefix, model
    for name, child in model._modules.items():
        if child is not None:
            child_prefix = prefix + '.' + name if prefix else name
            yield from _module_paths(child, child_prefix, ancestors + (id(model),))


def _aliases(model):
    groups = {}
    for prefix, module in _module_paths(model):
        for name, param in module._parameters.items():
            if param is not None:
                full = prefix + '.' + name if prefix else name
                groups.setdefault(id(param), []).append(full)
    return sorted(sorted(group) for group in groups.values() if len(group) > 1)


def _buffers(model):
    result = {}
    for prefix, module in model.named_modules():
        for name, tensor in module._buffers.items():
            full = prefix + '.' + name if prefix else name
            result[full] = (module, name, tensor, name not in module._non_persistent_buffers_set)
    return result


def _runtime_attributes(model):
    # HF4.28 RoPE may grow nonpersistent caches; restoring tensors alone is not enough.
    return {name: {'max_seq_len_cached': int(module.max_seq_len_cached)}
            for name, module in model.named_modules() if hasattr(module, 'max_seq_len_cached')}


def _config(model):
    if not (hasattr(model, 'config') and hasattr(model.config, 'to_dict')):
        return None
    # HF id2label uses integer keys; JSON manifests normalize those to strings.
    return json.loads(json.dumps(model.config.to_dict(), sort_keys=True))


def _buffer_aliases(model):
    groups = {}
    for name, (_, _, tensor, _) in _buffers(model).items():
        if tensor is not None:
            groups.setdefault(id(tensor), []).append(name)
    return sorted(sorted(group) for group in groups.values() if len(group) > 1)


def base_identity_from_prepared(prepared_dir):
    """Pin original source/tokenizer/base artifacts without rehashing 16GB shards.

    Large shard digests come from the pre-existing, published-SHA-verified fetch
    receipt plus current size/mtime checks. The actual loaded model is separately
    hashed by capture_reference(), so this is not a fresh full-file SHA audit.
    """
    prepared = Path(prepared_dir).resolve()
    spec_path = prepared / 'TRAINING_SPEC.json'
    spec = json.loads(spec_path.read_text())
    if prepared.name != 'prepared_full_s42_v2' or spec['seed'] != 42:
        raise ValueError('Original frozen baseline initialization required')
    receipt_path = Path(spec['checkpoint_fetch_receipt'])
    fetch = json.loads(receipt_path.read_text())
    if fetch['status'] != 'complete' or fetch['revision'] != spec['checkpoint_revision']:
        raise ValueError('Original checkpoint verification receipt is incomplete')
    records = []
    checkpoint = Path(spec['assets']['checkpoint'])
    for row in fetch['files']:
        path = checkpoint / row['file']
        if path.stat().st_size != row['bytes']:
            raise ValueError('Base artifact size changed: ' + row['file'])
        if row['file'].endswith('.bin'):
            if not row['published_sha256'] or row['published_sha256'] != row['verified_sha256']:
                raise ValueError('Base shard lacks published SHA verification')
            if path.stat().st_mtime > receipt_path.stat().st_mtime:
                raise ValueError('Base shard changed after prior verification')
            sha = row['verified_sha256']
        else:
            sha = file_sha256(path)
        records.append({'file': row['file'], 'bytes': row['bytes'], 'sha256': sha})
    sources = {}
    for row in spec['patched_sources'] + spec['source_snapshot']:
        actual = file_sha256(prepared / row['relative_path'])
        if actual != row['sha256']:
            raise ValueError('Frozen source changed: ' + row['relative_path'])
        sources[row['relative_path']] = actual
    identity = {'prepared': str(prepared), 'spec_sha256': file_sha256(spec_path),
                'checkpoint_revision': spec['checkpoint_revision'], 'checkpoint_files': records,
                'fetch_receipt_sha256': file_sha256(receipt_path), 'source_sha256': sources,
                'assets': spec['assets'], 'seed': 42,
                'clip_small_file_sha256': {name: file_sha256(Path(spec['assets']['clip']) / name)
                                          for name in ('config.json', 'preprocessor_config.json')},
                'runtime_versions': {name: importlib.metadata.version(name)
                                     for name in ('torch', 'transformers', 'peft', 'deepspeed')},
                'loader_sha256': file_sha256(Path(__file__).with_name('legion_text_adapter.py')),
                'file_verification': 'historical published SHA for large shards; current SHA for small assets/source',
                'loaded_model_verification': 'all parameter SHA captured separately after original loader/runtime dtype setup'}
    identity['sha256'] = _json_hash(identity)
    return identity


def capture_reference(model, base_identity, *, enforce_legion=True):
    """Capture ORIGINAL model just after final dtype/LoRA setup, before any update.

    This streams all parameter bytes through CPU (~16GB for LEGION) but writes no
    weight files. Rank zero only. Do not call on an already-trained model.
    """
    _rank_zero()
    model = _core(model)
    parameters = {name: dict(_tensor_row(param), trainable=param.requires_grad)
                  for name, param in _parameters(model).items()}
    trainable = [row for row in parameters.values() if row['trainable']]
    if enforce_legion and (len(trainable) != EXPECTED_TENSORS or sum(r['elements'] for r in trainable) != EXPECTED_ELEMENTS):
        raise ValueError('Not the complete original 297-tensor LEGION trainable contract')
    result = {'base_identity': base_identity, 'parameters': parameters, 'aliases': _aliases(model),
              'config': _config(model), 'enforce_legion': enforce_legion,
              'trainable_tensors': len(trainable), 'trainable_elements': sum(r['elements'] for r in trainable),
              'origin_capture_requirement': 'caller captured fresh original model before all optimizer updates'}
    result['original_model_parameters_sha256'] = _json_hash(parameters)
    return result


def _assert_layout(model, reference, *, check_dtype=True):
    parameters = _parameters(model)
    if set(parameters) != set(reference['parameters']) or _aliases(model) != reference['aliases']:
        raise ValueError('Parameter names or tied-parameter aliases changed')
    for name, param in parameters.items():
        row = reference['parameters'][name]
        if list(param.shape) != row['shape'] or bool(param.requires_grad) != row['trainable']:
            raise ValueError('Shape/trainable contract changed: ' + name)
        if check_dtype and str(param.dtype) != row['dtype']:
            raise ValueError('Parameter dtype changed: ' + name)
    if _config(model) != reference['config']:
        raise ValueError('Model config changed')
    return parameters


def save_delta(model, checkpoint_dir, reference, *, metadata=None, reserve_bytes=1536 * 1024**2):
    """Save all trainable tensors and all buffers. Never replaces an existing delta.

    Intended for model evaluation/reconstruction only; no optimizer/RNG/scheduler
    state is written, and this is explicitly NOT a resumable training checkpoint.
    """
    import torch
    _rank_zero()
    model = _core(model)
    target = Path(checkpoint_dir).resolve()
    if target.exists():
        raise FileExistsError(target)
    parameters = _assert_layout(model, reference)
    for name, param in parameters.items():
        if not param.requires_grad and tensor_sha256(param) != reference['parameters'][name]['sha256']:
            raise ValueError('Frozen parameter changed; delta cannot reconstruct it: ' + name)
    tensors = {'parameters': {}, 'buffers': {}}
    for name, param in parameters.items():
        if param.requires_grad:
            tensors['parameters'][name] = _tensor_copy(param)
    buffer_spec = {}
    for name, (_, _, tensor, persistent) in _buffers(model).items():
        buffer_spec[name] = {'persistent': persistent, 'tensor': tensor is not None}
        if tensor is not None:
            tensors['buffers'][name] = _tensor_copy(tensor)
    tensor_spec = {group: {name: _tensor_row(value) for name, value in rows.items()}
                   for group, rows in tensors.items()}
    estimated = sum(row['bytes'] for group in tensor_spec.values() for row in group.values())
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < estimated + reserve_bytes + 4 * 1024**2:
        raise OSError('Insufficient disk for one atomic delta plus explicit reserve')
    temporary = target.parent / ('.' + target.name + '.partial-' + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        weights = temporary / 'model_delta.pt'
        torch.save(tensors, weights)
        with weights.open('rb') as stream:
            os.fsync(stream.fileno())
        manifest = {'format': FORMAT, 'reference': reference, 'tensor_spec': tensor_spec,
                    'buffer_spec': buffer_spec, 'buffer_aliases': _buffer_aliases(model),
                    'runtime_attributes': _runtime_attributes(model),
                    'weights_file': weights.name, 'weights_sha256': file_sha256(weights),
                    'weights_bytes': weights.stat().st_size, 'payload_bytes': estimated,
                    'metadata': metadata or {}, 'optimizer_resume_supported': False,
                    'omitted_training_state': ['optimizer', 'FP32 master weights', 'scheduler', 'RNG', 'data cursor'],
                    'reconstruction_verified': False,
                    'verification_requirement': 'fresh original model, load_delta, then full byte-identity verification'}
        (temporary / 'MANIFEST.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
        with (temporary / 'MANIFEST.json').open('rb') as stream:
            os.fsync(stream.fileno())
        if target.exists():
            raise FileExistsError(target)
        os.rename(temporary, target)
    except BaseException:
        # Only our own uncommitted partial directory; preserve every prior checkpoint.
        shutil.rmtree(temporary)
        raise
    return manifest


def _dtype(name):
    import torch
    if not name.startswith('torch.') or not isinstance(getattr(torch, name[6:], None), torch.dtype):
        raise ValueError('Unknown tensor dtype: ' + name)
    return getattr(torch, name[6:])


def load_delta(model, checkpoint_dir, *, expected_base_identity, align_runtime_dtypes=False):
    """Reconstruct onto a FRESH original model and verify every parameter+buffer.

    Never call on an initialized optimizer: explicit dtype alignment replaces data
    storage. A model-only load cannot restore the omitted training state.
    """
    import torch
    model = _core(model)
    path = Path(checkpoint_dir)
    manifest = json.loads((path / 'MANIFEST.json').read_text())
    if manifest['format'] != FORMAT or manifest['optimizer_resume_supported']:
        raise ValueError('Unsupported checkpoint format')
    reference = manifest['reference']
    if reference['base_identity'] != expected_base_identity:
        raise ValueError('Wrong original base identity')
    parameters = _assert_layout(model, reference, check_dtype=not align_runtime_dtypes)
    if align_runtime_dtypes:
        # DeepSpeed BF16 setup may have cast LoRA FP32 from the author loader.
        # Restore that explicit runtime dtype before checking ORIGINAL bytes.
        for name, param in parameters.items():
            dtype = _dtype(reference['parameters'][name]['dtype'])
            if param.dtype != dtype:
                param.data = param.data.to(dtype=dtype)
    for name, param in parameters.items():
        if tensor_sha256(param) != reference['parameters'][name]['sha256']:
            raise ValueError('Fresh original parameter mismatch before applying delta: ' + name)
    weight_path = path / manifest['weights_file']
    if weight_path.name != 'model_delta.pt' or weight_path.parent.resolve() != path.resolve():
        raise ValueError('Invalid weight file path')
    if weight_path.stat().st_size != manifest['weights_bytes'] or file_sha256(weight_path) != manifest['weights_sha256']:
        raise ValueError('Checkpoint file hash/size mismatch')
    # These are locally created trusted tensor dictionaries; no arbitrary remote pickle.
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = True
    payload = torch.load(weight_path, **kwargs)
    if set(payload) != {'parameters', 'buffers'}:
        raise ValueError('Unexpected payload groups')
    for group in payload:
        if set(payload[group]) != set(manifest['tensor_spec'][group]):
            raise ValueError('Missing or extra delta tensors')
        for name, tensor in payload[group].items():
            if _tensor_row(tensor) != manifest['tensor_spec'][group][name]:
                raise ValueError('Tensor hash/shape/dtype mismatch: ' + name)
    expected_trainable = {name for name, param in parameters.items() if param.requires_grad}
    if set(payload['parameters']) != expected_trainable:
        raise ValueError('Delta does not contain ALL trainable parameters')
    buffers = _buffers(model)
    if set(buffers) != set(manifest['buffer_spec']):
        raise ValueError('Registered buffer names changed')
    with torch.no_grad():
        for name, tensor in payload['parameters'].items():
            parameters[name].copy_(tensor.to(device=parameters[name].device))
        for name, (_, _, _, persistent) in buffers.items():
            if persistent != manifest['buffer_spec'][name]['persistent']:
                raise ValueError('Buffer persistence changed: ' + name)
        for name, (module, leaf, old, _) in buffers.items():
            tensor = payload['buffers'].get(name)
            # Preserve registration and allow grown RoPE cache shapes to replace originals.
            device = old.device if old is not None else next(model.parameters()).device
            module._buffers[leaf] = tensor.to(device=device).clone() if tensor is not None else None
    for group in manifest['buffer_aliases']:
        first_module, first_leaf, _, _ = buffers[group[0]]
        for name in group[1:]:
            module, leaf, _, _ = buffers[name]
            module._buffers[leaf] = first_module._buffers[first_leaf]
    if _buffer_aliases(model) != manifest['buffer_aliases']:
        raise ValueError('Reconstructed buffer aliases differ')
    modules = dict(model.named_modules())
    for name, attrs in manifest['runtime_attributes'].items():
        if name not in modules or set(attrs) != {'max_seq_len_cached'}:
            raise ValueError('Unexpected runtime attribute')
        modules[name].max_seq_len_cached = attrs['max_seq_len_cached']
    # Verify ALL parameter bytes, not just LoRA or trainable subsets.
    for name, param in parameters.items():
        row = (manifest['tensor_spec']['parameters'][name] if param.requires_grad
               else reference['parameters'][name])
        if tensor_sha256(param) != row['sha256'] or str(param.dtype) != row['dtype']:
            raise ValueError('Reconstructed parameter differs: ' + name)
    for name, (_, _, tensor, _) in _buffers(model).items():
        if (tensor is not None) != manifest['buffer_spec'][name]['tensor']:
            raise ValueError('Reconstructed buffer presence differs: ' + name)
        if tensor is not None and _tensor_row(tensor) != manifest['tensor_spec']['buffers'][name]:
            raise ValueError('Reconstructed buffer differs: ' + name)
    receipt = {'status': 'all_parameter_and_buffer_bytes_verified', 'format': FORMAT,
               'checkpoint_weights_sha256': manifest['weights_sha256'],
               'original_model_parameters_sha256': reference['original_model_parameters_sha256'],
               'parameters_verified': len(parameters), 'trainable_tensors': len(expected_trainable),
               'buffers_verified': len(buffers), 'optimizer_resume_supported': False,
               'scope': 'exact model tensors/known RoPE cache attributes; evaluation mode and deterministic output probe are caller responsibilities'}
    return receipt


def save_latest_delta(model, arm_checkpoint_root, reference, *, metadata=None,
                      reserve_bytes=1536 * 1024**2, shared_lock_path=None):
    """Serialize arm saves and keep only its latest MODEL checkpoint.

    shared_lock_path must be the same for A and C; default is this module's folder.
    reserve_bytes must also include any future space already committed to another
    active job, not merely the 1.5GiB safety floor. No old checkpoint is removed
    until a complete newly checksummed delta exists. Initial save has no replacement.
    """
    import fcntl
    _rank_zero()
    if reserve_bytes < 1536 * 1024**2:
        raise ValueError('At least 1.5GiB shared filesystem reserve is required')
    root = Path(arm_checkpoint_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'latest'
    lock = Path(shared_lock_path) if shared_lock_path else Path(__file__).resolve().parent / '.delta_save.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        if any(root.glob('.previous-*')):
            raise RuntimeError('Interrupted prior commit; preserve both copies for manual recovery')
        previous_manifest = None
        if target.exists():
            old_manifest_path = target / 'MANIFEST.json'
            previous_manifest = json.loads(old_manifest_path.read_text())
            if (previous_manifest.get('format') != FORMAT or
                    previous_manifest['reference']['base_identity'] != reference['base_identity']):
                raise ValueError('Refusing to replace a checkpoint outside this delta/base format')
        candidate = root / ('.candidate-' + uuid.uuid4().hex)
        manifest = save_delta(model, candidate, reference, metadata=metadata, reserve_bytes=reserve_bytes)
        backup = root / ('.previous-' + uuid.uuid4().hex)
        try:
            if previous_manifest is not None:
                os.rename(target, backup)
            os.rename(candidate, target)
        except BaseException:
            if backup.exists() and not target.exists():
                os.rename(backup, target)
            raise
        # Small historical receipts survive, but the superseded model payload does not.
        if previous_manifest is not None:
            history = root / 'superseded'
            history.mkdir(exist_ok=True)
            (history / (previous_manifest['weights_sha256'] + '.json')).write_text(
                json.dumps({'weights_sha256': previous_manifest['weights_sha256'],
                            'weights_bytes': previous_manifest['weights_bytes'],
                            'metadata': previous_manifest['metadata'],
                            'replacement_sha256': manifest['weights_sha256']}, indent=2) + '\n')
            shutil.rmtree(backup)
        return dict(manifest, checkpoint_dir=str(target), global_save_lock=str(lock))
