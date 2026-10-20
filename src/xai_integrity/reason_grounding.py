"""Ground complete official explanation prediction states using the native SAM.

The author caption/SEG/explanation sequence is decoded ONCE. The existing text
projection is also evaluated once; complete, already supervised explanation
prediction states are pooled from that projection and sent to the SAME SAM
prompt encoder and mask decoder. Native CE/BCE/Dice and native SEG decoding are
untouched. No new parameters, image encodings, LM replay, quality head, or data.

This is explanation-state grounding regularization, not a guarantee of faithful
free generation. The selected state predicts its target token (does not read
that current target token), but does see teacher-forced previous caption and
explanation tokens. Region annotation is spatial supervision, not the truth or
validity of every explanation claim. The auxiliary query is absent at inference.
"""
import ast
from collections import Counter
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
from types import MethodType

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get('XAI_EXPLANATION_DATA', ROOT / 'explanation_data.py'))
DATA_SHA256 = 'e7fd81df89b29b604d7b79255eebc8315fbbdc21428ed4ed069e4bcec89de92b'
IMAGE_TOKEN = -200
SEG_TOKEN = 32004
IMAGE_PATCH_TOKENS = 576
PREDICTION_POSITION_SHIFT = IMAGE_PATCH_TOKENS - 2  # image expansion575, next-token -1
API = 'prepare_native_forward_reason_grounding_one_extra_decoder_v1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if _sha(DATA_PATH) != DATA_SHA256:
    raise ValueError('Frozen official explanation provenance dependency changed')
_spec = importlib.util.spec_from_file_location('reason_grounding_official_provenance', DATA_PATH)
data = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = data
_spec.loader.exec_module(data)


def load_author_mask_losses(source_dir):
    """Load exactly the author's two pure loss definitions without heavy imports."""
    path = Path(source_dir) / 'model/Legion.py'
    tree = ast.parse(path.read_text())
    names = ('compute_sigmoid_cross_entropy', 'calculate_dice_loss')
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {fn.name for fn in functions} != set(names) or len(functions) != 2:
        raise ValueError('Expected exact author BCE/Dice definitions')
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[names[0]], namespace[names[1]], _sha(path)


def explanation_prediction_positions(begin, end, *, image_index, retained_tokens,
                                     projected_tokens, device=None):
    """Positions predicting ids[begin:end], in native expanded LM coordinates.

    A text token after the one image is shifted by575 in expanded coordinates;
    its predictive hidden state is one position earlier. No current target
    token representation is included in that token's selected predictive state.
    """
    if not (0 <= image_index < begin < end <= retained_tokens):
        raise data.Ineligible('invalid_complete_explanation_after_image')
    if projected_tokens != retained_tokens + IMAGE_PATCH_TOKENS - 1:
        raise data.Ineligible('native_expanded_hidden_length_mismatch')
    positions = torch.arange(begin + PREDICTION_POSITION_SHIFT,
                             end + PREDICTION_POSITION_SHIFT, device=device)
    if int(positions[-1]) >= projected_tokens:
        raise data.Ineligible('explanation_prediction_positions_outside_native_hidden')
    return positions


def _parameter_contract(core):
    return tuple((name, id(p), tuple(p.shape), p.requires_grad) for name, p in core.named_parameters())


class ReasonGroundingController:
    def __init__(self, core, *, tokenizer, annotation_path, annotation_sha256,
                 source_dir, coefficient=.1, max_expanded_length=1536,
                 strict_author=True):
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError('Finite nonnegative auxiliary coefficient required')
        if max_expanded_length != 1536:
            raise ValueError('Original author1536 context required')
        if getattr(core, '_official_reason_grounding_installed', False):
            raise ValueError('Reason grounding already installed')
        if getattr(core, '_official_insight_installed', False):
            raise ValueError('Reason grounding must not be stacked on response replay')
        if [tokenizer.convert_tokens_to_ids(t) for t in ('<p>', '</p>', '[SEG]')] != [32005, 32006, SEG_TOKEN]:
            raise ValueError('Original tokenizer tag IDs required')
        for method in ('_generate_and_postprocess_masks', 'forward'):
            if not callable(getattr(core, method, None)):
                raise ValueError('Missing native LEGION method: ' + method)
        if not hasattr(core.model, 'text_hidden_fcs') or len(core.model.text_hidden_fcs) != 1:
            raise ValueError('Exactly one existing native text projection required')
        if float(core.bce_loss_weight) != .4 or float(core.dice_loss_weight) != .2:
            raise ValueError('Original author BCE=.4/Dice=.2 weights required')
        if strict_author:
            trainable = [p for p in core.parameters() if p.requires_grad]
            if core.__class__.__name__ != 'LegionForCausalLM' or len(trainable) != 297 or sum(p.numel() for p in trainable) != 587427044:
                raise ValueError('Original LEGION/297 trainables required')
            if core.config.use_cache:
                raise ValueError('Native use_cache=False required')
            if any(p.requires_grad for p in core.model.grounding_encoder.image_encoder.parameters()):
                raise ValueError('Frozen original SAM image encoder required')
        self.core = core
        self.tokenizer = tokenizer
        self.coefficient = float(coefficient)
        self.max_expanded_length = max_expanded_length
        self.strict_author = strict_author
        self.conv, self.token_image, self.allowed_questions = data.load_author_helpers(source_dir)
        self.index = data.OfficialTrainIndex(annotation_path, annotation_sha256, strict=strict_author)
        self.bce, self.dice, self.author_model_sha256 = load_author_mask_losses(source_dir)
        self.context = None
        self.last_receipt = None
        self.auxiliary_gradient = {'hook_calls': 0, 'nonzero_calls': 0,
                                   'finite': True, 'max_abs': 0.,
                                   'requires_grad': False}
        self.parameter_contract = _parameter_contract(core)
        self.had_forward = 'forward' in core.__dict__
        self.own_forward = core.__dict__.get('forward')
        self.original_forward = core.forward
        self.original_generate_masks = core._generate_and_postprocess_masks
        controller = self

        def dispatch(this, *args, **kwargs):
            return controller._forward(args, kwargs)

        core.forward = MethodType(dispatch, core)
        core._official_reason_grounding_installed = True
        if _parameter_contract(core) != self.parameter_contract:
            raise RuntimeError('Installation changed model parameter contract')

    def prepare(self, batch, optimizer_update_1based=None, micro_0based=None, **kwargs):
        """Bind every complete ref in THIS native batch; never alter its tokens/GT.

        Variable auxiliary ref counts need no extra backward/LM invocation or
        global eligibility gate. Each rank still executes one original backward.
        A malformed/truncated mask count fails closed for the entire auxiliary.
        A missing/incomplete individual explanation excludes only that ref.
        """
        if kwargs:
            raise TypeError('Unknown prepare options: ' + ','.join(kwargs))
        if self.context is not None:
            raise RuntimeError('Previous prepared native batch was not consumed')
        receipt = {'api': API, 'method_variant': 'official_reason_grounding',
                   'scheduled': True, 'eligible': False, 'completed': False,
                   'optimizer_update': optimizer_update_1based, 'micro': micro_0based,
                   'coefficient': self.coefficient, 'bce_weight': .4, 'dice_weight': .2,
                   'extra_LM_forwards': 0, 'extra_CLIP_forwards': 0,
                   'extra_SAM_encoder_forwards': 0, 'extra_decoder_forwards': 0,
                   'extra_prompt_encoder_forwards': 0,
                   'extra_backward_calls': 0, 'extra_projection_forwards': 0,
                   'extra_data': False, 'skip_reason': None,
                   'additional_GT_mask_exposures': 0,
                   'additional_GT_pixel_exposures': 0,
                   'same_supervision_quantity': False,
                   'supervision_scope': 'official referent masks for existing complete supervised explanation prediction states; not explanation validity',
                   'teacher_forced_previous_text': True,
                   'current_target_token_hidden_read': False,
                   'generation_faithfulness_proven': False}
        self.last_receipt = receipt
        started = time.monotonic()
        try:
            if self.coefficient == 0:
                raise data.Ineligible('zero_coefficient_exact_native_path')
            if batch.get('inference', False) or not torch.is_grad_enabled():
                raise data.Ineligible('inference_or_no_grad_exact_native_path')
            record, question, gt, binding = self.index.join(
                batch, tokenizer=self.tokenizer, conv_template=self.conv,
                token_image=self.token_image, allowed_questions=self.allowed_questions,
                max_expanded_length=self.max_expanded_length, strict=self.strict_author)
            ids = batch['input_ids'][0].detach().cpu()
            if int((ids == IMAGE_TOKEN).sum()) != 1:
                raise data.Ineligible('expected_one_native_image_placeholder')
            image_index = int(torch.nonzero(ids == IMAGE_TOKEN).flatten()[0])
            seg_positions = torch.nonzero(ids == SEG_TOKEN).flatten()
            # The author loss truncates GT to predicted SEG masks when token
            # truncation occurs. Preserve that native behavior, but do NOT turn
            # the remainder into an apparently valid auxiliary mask join.
            if len(seg_positions) != len(gt) or len(gt) != len(record['refs']):
                raise data.Ineligible('native_SEG_GT_ref_count_mismatch_no_aux_truncation')
            legal = []
            excluded = []
            phrase_counts = Counter(ref['phrase'].strip().casefold() for ref in record['refs'])
            explanation_counts = Counter(ref['explanation'].strip().casefold() for ref in record['refs']
                                         if isinstance(ref.get('explanation'), str) and ref['explanation'].strip())
            for mask_index, ref in enumerate(record['refs']):
                try:
                    if phrase_counts[ref['phrase'].strip().casefold()] != 1:
                        raise data.Ineligible('ambiguous_repeated_official_referent')
                    explanation = ref.get('explanation')
                    if isinstance(explanation, str) and explanation.strip() and explanation_counts[explanation.strip().casefold()] != 1:
                        raise data.Ineligible('ambiguous_repeated_official_explanation')
                    if any(control in ref['phrase'] or (isinstance(explanation, str) and control in explanation)
                           for control in data.SPECIAL_STRINGS):
                        raise data.Ineligible('official_ref_or_explanation_has_control_string')
                    native = data.native_supervision_binding(
                        record, ref, mask_index, batch, question=question,
                        tokenizer=self.tokenizer, conv_template=self.conv,
                        token_image=self.token_image)
                    begin, end = native['native_explanation_token_begin'], native['native_explanation_token_end']
                    following_markers = torch.nonzero((ids == 32005) &
                                                     (torch.arange(len(ids)) > native['native_ref_SEG_position'])).flatten()
                    if len(following_markers) and end > int(following_markers[0]):
                        raise data.Ineligible('official_explanation_outside_own_ref_before_next_marker')
                    if any(int(token) in (IMAGE_TOKEN, 32005, 32006, SEG_TOKEN) for token in ids[begin:end]):
                        raise data.Ineligible('official_explanation_span_contains_ref_or_control_token')
                    explanation_prediction_positions(begin, end, image_index=image_index,
                        retained_tokens=len(ids), projected_tokens=len(ids) + 575)
                    legal.append({'mask_index': mask_index, 'original_ref_index': ref['original_ref_index'],
                                  'begin': begin, 'end': end, 'binding': native})
                except data.Ineligible as exc:
                    excluded.append({'original_ref_index': ref['original_ref_index'], 'reason': str(exc)})
            overlaps = set()
            for position, first in enumerate(legal):
                for second in legal[position + 1:]:
                    if max(first['begin'], second['begin']) < min(first['end'], second['end']):
                        overlaps.update((first['mask_index'], second['mask_index']))
            if overlaps:
                for ref in legal:
                    if ref['mask_index'] in overlaps:
                        excluded.append({'original_ref_index': ref['original_ref_index'],
                                         'reason': 'overlapping_official_explanation_token_spans'})
                legal = [ref for ref in legal if ref['mask_index'] not in overlaps]
            receipt.update(**binding, source=str(batch['image_paths'][0]),
                           official_ref_count=len(record['refs']), legal_ref_count=len(legal),
                           excluded_refs=excluded, native_retained_tokens=len(ids),
                           native_expanded_tokens=len(ids) + 575,
                           target_scope='all_complete_native_supervised_official_explanations',
                           selected_refs=[dict(mask_index=x['mask_index'], original_ref_index=x['original_ref_index'],
                                              **x['binding']) for x in legal],
                           author_model_sha256=self.author_model_sha256,
                           provenance_source_sha256=DATA_SHA256)
            if not legal:
                raise data.Ineligible('no_complete_native_supervised_official_explanation')
            self.context = {'legal': legal, 'image_index': image_index,
                            'ids': ids.clone(), 'gt': batch['masks_list'][0],
                            'resize_list': batch['resize_list'], 'label_list': batch['label_list']}
            receipt['eligible'] = True
            return receipt
        except data.Ineligible as exc:
            receipt['skip_reason'] = str(exc)
            return receipt
        finally:
            receipt['prepare_seconds'] = time.monotonic() - started

    def _forward(self, args, kwargs):
        context, self.context = self.context, None
        receipt = self.last_receipt
        # No hooks and no alternative loss path for zero/no_grad/inference.
        if context is None or self.coefficient == 0 or not torch.is_grad_enabled() or kwargs.get('inference', False) or 'past_key_values' in kwargs:
            if context is not None:
                receipt['skip_reason'] = 'inference_or_no_grad_exact_native_path'
            return self.original_forward(*args, **kwargs)
        if args:
            raise ValueError('Native LEGION uses named batch arguments only')
        if not torch.equal(kwargs['input_ids'][0].detach().cpu(), context['ids']):
            raise ValueError('Prepared native token row differs from forward')
        if kwargs.get('masks_list') is None or len(kwargs['masks_list']) != 1 or kwargs['masks_list'][0] is not context['gt']:
            raise ValueError('Prepared native mask tensor differs from forward')
        captured = {'projection_calls': 0, 'native_mask_calls': 0}

        def projection_hook(module, inputs, projected):
            captured['projection_calls'] += 1
            captured['projected'] = projected  # keep original differentiable graph

        original_masks = self.core._generate_and_postprocess_masks
        had_masks = '_generate_and_postprocess_masks' in self.core.__dict__
        own_masks = self.core.__dict__.get('_generate_and_postprocess_masks')

        def capture_masks(this, pred_embeddings, image_embeddings, resize_list, label_list, infer=False):
            captured['native_mask_calls'] += 1
            value = original_masks(pred_embeddings, image_embeddings, resize_list, label_list, infer=infer)
            captured.update(image_embeddings=image_embeddings, native_pred_masks=value)
            return value

        hook = self.core.model.text_hidden_fcs[0].register_forward_hook(projection_hook)
        self.core._generate_and_postprocess_masks = MethodType(capture_masks, self.core)
        try:
            output = self.original_forward(*args, **kwargs)
        finally:
            hook.remove()
            if had_masks:
                self.core._generate_and_postprocess_masks = own_masks
            else:
                delattr(self.core, '_generate_and_postprocess_masks')
        if not isinstance(output, dict) or 'loss' not in output:
            raise ValueError('Expected original native training loss dictionary')
        if captured['projection_calls'] != 1 or captured['native_mask_calls'] != 1:
            raise RuntimeError('Expected one original projection and native SAM decoding path')
        projected = captured['projected']
        embeddings = captured['image_embeddings']
        native_masks = captured['native_pred_masks']
        gt = context['gt']
        if not isinstance(projected, torch.Tensor) or projected.ndim != 3 or projected.shape[0] != 1 or projected.shape[-1] != 256:
            raise ValueError('Expected native projected hidden tensor [1,T,256]')
        if not isinstance(embeddings, torch.Tensor) or embeddings.shape[0] != 1 or embeddings.requires_grad:
            raise ValueError('Expected one original frozen SAM image embedding')
        receipt.update(native_projection_calls=captured['projection_calls'], native_mask_calls=captured['native_mask_calls'],
                       native_projected_shape=list(projected.shape), native_GT_values_preserved=True)
        if len(native_masks) != 1 or native_masks[0].shape != gt.shape:
            receipt['skip_reason'] = 'native_pred_GT_shape_mismatch_no_aux_truncation'
            receipt['eligible'] = False
            return output
        queries = []
        for ref in context['legal']:
            positions = explanation_prediction_positions(ref['begin'], ref['end'],
                image_index=context['image_index'], retained_tokens=len(context['ids']),
                projected_tokens=projected.shape[1], device=projected.device)
            queries.append(projected[0].index_select(0, positions).mean(0))
        queries = torch.stack(queries)
        receipt['auxiliary_query_requires_grad'] = bool(queries.requires_grad)
        self.auxiliary_gradient['requires_grad'] = bool(queries.requires_grad)
        if queries.requires_grad:
            # This tensor is used ONLY by the auxiliary decoder, so its hook
            # cannot count a native-loss gradient as an auxiliary witness.
            def auxiliary_query_gradient(gradient):
                detached = gradient.detach()
                finite = bool(torch.isfinite(detached).all())
                maximum = float(detached.abs().max())
                nonzero = bool(maximum > 0)
                self.auxiliary_gradient['hook_calls'] += 1
                self.auxiliary_gradient['nonzero_calls'] += int(nonzero)
                self.auxiliary_gradient['finite'] &= finite
                self.auxiliary_gradient['max_abs'] = max(self.auxiliary_gradient['max_abs'], maximum)
                receipt['auxiliary_query_gradient'] = {'finite': finite, 'max_abs': maximum, 'nonzero': nonzero}
            queries.register_hook(auxiliary_query_gradient)
        started = time.monotonic()
        # Hooks were removed already: use original helper without changing SEG
        # prompt count/batching. All auxiliary refs form one decoder call/image.
        def count_decoder(module, inputs, output):
            receipt['extra_decoder_forwards'] += 1
        def count_prompt_encoder(module, inputs, output):
            receipt['extra_prompt_encoder_forwards'] += 1
        grounding = self.core.model.grounding_encoder
        decoder_hook = grounding.mask_decoder.register_forward_hook(count_decoder)
        prompt_hook = grounding.prompt_encoder.register_forward_hook(count_prompt_encoder)
        cpu_rng = torch.get_rng_state().clone()
        cuda_devices = [queries.device.index if queries.device.index is not None else torch.cuda.current_device()] if queries.device.type == 'cuda' else []
        cuda_rng = {device: torch.cuda.get_rng_state(device).clone() for device in cuda_devices}
        try:
            auxiliary_masks = self.original_generate_masks(
                [queries], embeddings, context['resize_list'], context['label_list'])
        finally:
            decoder_hook.remove()
            prompt_hook.remove()
        cpu_unchanged = torch.equal(cpu_rng, torch.get_rng_state())
        cuda_unchanged = all(torch.equal(state, torch.cuda.get_rng_state(device)) for device, state in cuda_rng.items())
        receipt.update(auxiliary_rng_cpu_unchanged=cpu_unchanged,
                       auxiliary_rng_cuda_unchanged=cuda_unchanged,
                       auxiliary_rng_unchanged=cpu_unchanged and cuda_unchanged)
        if not cpu_unchanged or not cuda_unchanged:
            raise RuntimeError('Auxiliary shared SAM decoder changed future native RNG; no hidden restoration')
        if receipt['extra_decoder_forwards'] != 1 or receipt['extra_prompt_encoder_forwards'] != 1:
            raise RuntimeError('Expected exactly one extra SAM prompt/decoder invocation per image')
        if len(auxiliary_masks) != 1:
            raise RuntimeError('Expected one auxiliary prediction mask tensor')
        indices = torch.tensor([ref['mask_index'] for ref in context['legal']], device=gt.device)
        target = gt.index_select(0, indices)
        prediction = auxiliary_masks[0]
        if prediction.shape != target.shape:
            raise RuntimeError('Auxiliary official ref/GT mask shape mismatch')
        count = len(context['legal'])
        receipt.update(additional_GT_mask_exposures=count,
                       additional_GT_pixel_exposures=target.numel(),
                       auxiliary_GT_max=float(target.detach().max()),
                       auxiliary_GT_nonbinary_pixels=int(((target != 0) & (target != 1)).sum()))
        bce = self.bce(prediction, target, mask_count=count)
        dice = self.dice(prediction, target, mask_count=count)
        auxiliary_loss = self.coefficient * (.4 * bce + .2 * dice)
        if not bool(torch.isfinite(auxiliary_loss.detach())):
            raise FloatingPointError('Nonfinite reason grounding loss')
        result = dict(output)
        result['loss'] = output['loss'] + auxiliary_loss
        result.update(reason_bce_loss=bce, reason_dice_loss=dice,
                      reason_grounding_loss=auxiliary_loss,
                      reason_count=torch.tensor(count, device=auxiliary_loss.device))
        receipt.update(completed=True, auxiliary_mask_count=count,
                       auxiliary_query_shape=list(queries.shape),
                       auxiliary_prediction_shape=list(prediction.shape),
                       auxiliary_bce=float(bce.detach()), auxiliary_dice=float(dice.detach()),
                       auxiliary_loss=float(auxiliary_loss.detach()),
                       auxiliary_decoder_seconds=time.monotonic() - started)
        return result

    def skip(self, reason):
        self.context = None
        if self.last_receipt is not None:
            self.last_receipt['skip_reason'] = reason
            self.last_receipt['eligible'] = False
        return self.last_receipt

    def remove(self):
        self.context = None
        if self.had_forward:
            self.core.forward = self.own_forward
        else:
            delattr(self.core, 'forward')
        if hasattr(self.core, '_official_reason_grounding_installed'):
            delattr(self.core, '_official_reason_grounding_installed')
        if _parameter_contract(self.core) != self.parameter_contract:
            raise RuntimeError('Model parameter contract changed while reason grounding was installed')


def install_reason_grounding(core, **kwargs):
    return ReasonGroundingController(core, **kwargs)
