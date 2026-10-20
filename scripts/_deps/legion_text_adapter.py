"""Independent text-only calls on the existing LEGION/LoRA parameters.

Importing this module does not load weights, allocate CUDA, or change any model.
Install only on a fresh experiment instance, never an active LEGION/C2 instance.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from types import MethodType

TEXT_FLAG = 'evidence_text_only'
EXPECTED_TENSORS = 297
EXPECTED_ELEMENTS = 587427044


def _canonical_name(name):
    while name.startswith('module.') or name.startswith('base_model.model.'):
        name = name[len('module.'):] if name.startswith('module.') else name[len('base_model.model.'):]
    return name


def unwrap_legion(model):
    """Locate the actual LM for installation, while later calls retain outer wrappers."""
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from peft import PeftModel
    seen = set()
    while not isinstance(model, LlamaForCausalLM):
        if id(model) in seen:
            raise TypeError('Cyclic or unsupported model wrapper')
        seen.add(id(model))
        if isinstance(model, PeftModel):
            if model.active_peft_config.is_prompt_learning:
                raise TypeError('Prefix/prompt tuning can carry hidden state; only existing LoRA is supported')
            model = model.get_base_model()
        elif type(model).__name__ in ('DeepSpeedEngine', 'DistributedDataParallel', 'DataParallel'):
            model = model.module
        else:
            raise TypeError('Unsupported wrapper: '+type(model).__name__)
    if not any(c.__name__ == 'LegionForCausalLM' for c in type(model).__mro__):
        raise TypeError('Expected original LEGION LM, not a replacement Llama model')
    return model


def trainable_signature(model):
    core = unwrap_legion(model)
    return sorted([{'name':_canonical_name(n),'shape':list(p.shape),'elements':p.numel(),'dtype':str(p.dtype)}
                   for n,p in core.named_parameters() if p.requires_grad],key=lambda r:r['name'])


def assert_trainable_contract(model, witness_path=None):
    rows = trainable_signature(model)
    assert len(rows) == EXPECTED_TENSORS, len(rows)
    assert sum(r['elements'] for r in rows) == EXPECTED_ELEMENTS
    if witness_path is not None:
        expected = json.loads(Path(witness_path).read_text())['parameters']
        compact = lambda records: sorted((_canonical_name(r['name']),tuple(r['shape']),r['elements']) for r in records)
        assert compact(rows) == compact(expected), 'Trainable names/shapes differ from frozen baseline witness'
    return {'trainable_tensors':len(rows),'trainable_elements':sum(r['elements'] for r in rows),
            'signature_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}


def install_text_mode(model, *, check_author_contract=True, witness_path=None):
    """Add an explicit dispatch without adding/changing any parameter or LoRA module.

    Call through model/engine afterwards, so DeepSpeed/DDP forward hooks remain active.
    This is a model-instance patch. It does not edit author source or global classes.
    """
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel
    core = unwrap_legion(model)
    before = trainable_signature(core)
    if check_author_contract:
        assert_trainable_contract(core,witness_path)
    if type(core.model).forward is not LlamaModel.forward:
        raise TypeError('Decoder forward differs from the verified native LlamaModel path')
    lora = [(n,m) for n,m in core.named_modules() if hasattr(m,'lora_A') and hasattr(m,'lora_B')]
    assert lora, 'No injected LoRA modules; refusing a base-only text call'
    assert all(not m.merged and not m.disable_adapters for _,m in lora), 'LoRA must remain active and unmerged'
    if getattr(core,'_evidence_text_mode_installed',False):
        return {'installed':True,'already_installed':True,'lora_modules':len(lora)}
    original_forward = core.forward

    def dispatch(self,*args,**kwargs):
        if not kwargs.pop(TEXT_FLAG,False):
            return original_forward(*args,**kwargs)
        if args:
            raise TypeError('Text-only branch accepts named token inputs only')
        # PEFT 0.5 forwards these absent optional arguments as explicit None.
        for name in ('inputs_embeds','past_key_values','output_attentions','output_hidden_states','return_dict'):
            if name in kwargs:
                value = kwargs.pop(name)
                if value is not None:
                    raise TypeError('Independent text call rejects '+name)
        allowed = {'input_ids','attention_mask','labels','position_ids'}
        unexpected = set(kwargs)-allowed
        if unexpected:
            raise TypeError('Independent text call rejects: '+str(sorted(unexpected)))
        ids = kwargs.get('input_ids')
        mask = kwargs.get('attention_mask')
        if ids is None or ids.ndim != 2 or mask is None or tuple(mask.shape) != tuple(ids.shape):
            raise ValueError('Explicit BxT input_ids and equal-shape attention_mask required')
        if ids.numel() == 0 or ids.min().item() < 0 or ids.max().item() >= self.config.vocab_size:
            raise ValueError('Invalid text tokens, including negative visual placeholders')
        if not mask.bool().any(dim=1).all().item():
            raise ValueError('Each text row must contain an attended token')
        # Direct class call uses self.model's CURRENT q/v LoRA layers and self.lm_head.
        # No multimodal prepare, visual encoder, SAM, SEG mask, or inherited KV state.
        result = LlamaForCausalLM.forward(self, **kwargs, use_cache=False,
                    output_attentions=False, output_hidden_states=False, return_dict=True)
        assert result.past_key_values is None and result.hidden_states is None
        return result

    core.forward = MethodType(dispatch,core)
    core._evidence_text_mode_installed = True
    assert before == trainable_signature(core), 'Installing text mode changed parameter contract'
    return {'installed':True,'already_installed':False,'lora_modules':len(lora),
            'parameter_contract_unchanged':True,'native_forward':'transformers4.28 LlamaForCausalLM.forward',
            'kv_cache':False,'visual_path':False,'new_parameters':0}


def text_forward(model, input_ids, attention_mask, labels=None, position_ids=None):
    """Return native CausalLMOutputWithPast; logits/loss retain autograd.

    model may be the original object, PEFT wrapper, or initialized DeepSpeed engine.
    install_text_mode must have run before this call. No image or hidden-state API.
    """
    core = unwrap_legion(model)
    if not getattr(core,'_evidence_text_mode_installed',False):
        raise RuntimeError('Install text mode on this fresh model first')
    return model(**{TEXT_FLAG:True},input_ids=input_ids,attention_mask=attention_mask,
                 labels=labels,position_ids=position_ids)


def selected_next_token_logits(output, attention_mask, vocabulary_ids):
    """Select existing lm_head columns at last attended position; no new head."""
    import torch
    logits = output.logits
    assert logits.ndim == 3 and tuple(logits.shape[:2]) == tuple(attention_mask.shape)
    positions = torch.arange(logits.shape[1],device=logits.device).expand(logits.shape[0],-1)
    last = positions.masked_fill(~attention_mask.bool(),-1).max(dim=1).values
    assert (last >= 0).all().item()
    vocab = torch.as_tensor(vocabulary_ids,dtype=torch.long,device=logits.device)
    assert vocab.ndim == 1 and vocab.numel() > 0
    return logits[torch.arange(logits.shape[0],device=logits.device),last].index_select(-1,vocab)


def load_original_author_model(prepared_dir, receipt_dir, *, local_rank=0):
    """GPU-worker-only loader reusing original author initialization, not trained weights.

    The author preparer moves its vision tower to local_rank. Do not invoke this
    function during CPU-only audits. No optimizer, checkpoint resume, or dataset is loaded.
    """
    import numpy as np
    import torch
    prepared = Path(prepared_dir).resolve()
    spec = json.loads((prepared/'TRAINING_SPEC.json').read_text())
    argv = list(spec['author_arguments']) + ['--local_rank',str(local_rank)]
    expected_origin = os.environ.get('XAI_GLaMM_ORIGIN', spec.get('assets', {}).get('checkpoint', ''))
    assert argv[argv.index('--version')+1] == expected_origin
    assert '--pretrained' in argv and spec['seed'] == 42
    assert prepared.name, 'Prepared model bundle path is required'
    receipt = Path(receipt_dir).resolve()
    assert receipt != prepared/'receipts' and not receipt.is_relative_to(prepared)
    receipt.mkdir(parents=True,exist_ok=True)
    os.environ['FORMAL_LEGION_RECEIPTS'] = str(receipt)
    sys.path[:0] = [str(prepared/'source'),str(prepared)]
    mp = prepared/'source/scripts/loc_exp/train.py'
    modspec = importlib.util.spec_from_file_location('response_original_author_train',mp)
    train = importlib.util.module_from_spec(modspec)
    modspec.loader.exec_module(train)
    cfg = train.parse_args(argv)
    cfg.formal_canary = False
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    tokenizer = train.setup_tokenizer_and_special_tokens(cfg)
    assert len(tokenizer) == 32007 and cfg.seg_token_idx == 32004
    model = train.initialize_model(cfg,tokenizer)
    # Author ignores the PEFT wrapper return; PEFT0.5 mutates q/v modules in place.
    train.prepare_model_for_training(model,tokenizer,cfg)
    contract = assert_trainable_contract(model,prepared/'receipts/trainable_rank0.json')
    installed = install_text_mode(model,witness_path=prepared/'receipts/trainable_rank0.json')
    return model,tokenizer,cfg,{'origin':expected_origin,'seed':42,**contract,**installed}
