"""Unchanged native-output capture and exact witness helpers. No q/verifier calls."""
def capture_native(model, tokenizer, author_call):
    """Observe the unmodified author call; retain IDs/masks before official saving."""
    import torch
    if model.training:
        raise ValueError('Native inference requires eval mode')
    old, existed, captured = model.evaluate, 'evaluate' in model.__dict__, []

    def tap(*args, **kwargs):
        ids, masks = old(*args, **kwargs)
        if ids.ndim != 2 or ids.shape[0] != 1 or len(masks) != 1:
            raise ValueError('Expected exactly one native image output')
        captured.append((ids[0].detach().cpu().clone(), masks[0].detach().cpu().clone()))
        return ids, masks

    model.evaluate = tap
    try:
        with torch.no_grad():
            text, masks, phrases = author_call(model)
    finally:
        if existed:
            model.evaluate = old
        else:
            delattr(model, 'evaluate')
    if len(captured) != 1:
        raise ValueError('Expected one native evaluate call')
    ids, mask_cpu = captured[0]
    raw = tokenizer.decode(ids[ids != -200], skip_special_tokens=False)
    raw = raw.replace('\n', '').replace('  ', ' ').split('ASSISTANT: ')[-1]
    return {'ids': ids.tolist(), 'raw': raw, 'text': text, 'phrases': list(phrases), 'masks': mask_cpu}


def compare_outputs(a, b, seg_token_idx):
    import torch
    same_shape = a['masks'].shape == b['masks'].shape
    equal = (same_shape and a['masks'].dtype == b['masks'].dtype and
             torch.equal(a['masks'].contiguous().view(torch.uint8), b['masks'].contiguous().view(torch.uint8)))
    binary_equal = same_shape and torch.equal(a['masks'] > 0, b['masks'] > 0)
    finite = bool(torch.isfinite(a['masks']).all() and torch.isfinite(b['masks']).all())
    return {'token_ids_equal': a['ids'] == b['ids'], 'raw_text_equal': a['raw'] == b['raw'],
            'text_equal': a['text'] == b['text'], 'phrases_equal': a['phrases'] == b['phrases'],
            'seg_count_equal': a['ids'].count(seg_token_idx) == b['ids'].count(seg_token_idx),
            'seg_count': a['ids'].count(seg_token_idx), 'mask_shape_equal': same_shape,
            'mask_shapes': [list(a['masks'].shape), list(b['masks'].shape)],
            'binary_masks_equal': bool(binary_equal), 'masks_bitwise_equal': bool(equal),
            'finite_masks': finite,
            'mask_max_abs_error': float((a['masks'].float() - b['masks'].float()).abs().max())
                if same_shape and a['masks'].numel() else (0.0 if same_shape else None),
            'binary_mask_pixels_different': int(((a['masks'] > 0) != (b['masks'] > 0)).sum())
                if same_shape else None}


def runtime_config(model):
    generation = model.generation_config
    if model.config.use_cache or generation.use_cache or generation.do_sample:
        raise ValueError('Unexpected native cache/sampling config; do not silently modify protocol')
    return {'model_use_cache': model.config.use_cache, 'generation_use_cache': generation.use_cache,
            'generation_config': generation.to_dict(), 'model_eval': not model.training}


