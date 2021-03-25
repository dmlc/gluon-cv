"""Utils for deep learning framework related functions"""

__all__ = ['_suggest_load_context']

def _suggest_load_context(model, mode, orig_ctx):
    """Get the correct context given the mode"""
    if not isinstance(orig_ctx, (list, tuple)):
        orig_ctx = [orig_ctx]
    try:
        import mxnet as mx
    except ImportError:
        mx = None
    try:
        import torch
    except ImportError:
        torch = None
    if mx is not None and isinstance(model, mx.gluon.Block):
        if mode == 'auto':
            if orig_ctx[0].device_type == 'gpu':
                mode = 'gpu'
            else:
                mode = 'cpu'
        if mode == 'cpu':
            return [mx.cpu()]
        if mode == 'gpu':
            return [mx.gpu(i) for i in range(mx.context.num_gpus())]
        if isinstance(mode, (list, tuple)):
            if not all([isinstance(i, int) for i in mode]):
                raise ValueError('Requires integer gpu id, given {}'.format(mode))
            return [mx.gpu(i) for i in mode if i in range(mx.context.num_gpus())]
    if torch is not None and isinstance(model, torch.Module):
        pass
    return None
