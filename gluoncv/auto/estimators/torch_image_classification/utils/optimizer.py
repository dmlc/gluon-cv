# pylint: disable=wildcard-import
from timm.optim.optim_factory import *

_TIMM_FC_NAMES = ('fc', 'head', 'last_linear', 'classifier')

def optimizer_kwargs(cfg):
    kwargs = dict(
        optimizer_name=cfg.optimizer.opt,
        learning_rate=cfg.train.lr,
        weight_decay=cfg.optimizer.weight_decay,
        momentum=cfg.optimizer.momentum)
    if cfg.optimizer.opt_eps is not None:
        kwargs['eps'] = cfg.optimizer.opt_eps
    if cfg.optimizer.opt_betas is not None:
        kwargs['betas'] = cfg.optimizer.opt_betas
    return kwargs

def add_weight_decay_filter_fc(model, learning_rate=0.001, weight_decay=1e-5, skip_list=(),
                               fc_names=_TIMM_FC_NAMES, feature_lr_mult=0.01, fc_lr_mult=1):
    if feature_lr_mult == 1 and fc_lr_mult == 1:
        return add_weight_decay(model, weight_decay, skip_list)
    decay_feat = []
    no_decay_feat = []
    decay_fc = []
    no_decay_fc = []
    for name, param in model.named_parameters():
        is_fc = False
        if any([name.startswith(pattern) for pattern in fc_names]):
            is_fc = True
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            if is_fc:
                no_decay_fc.append(param)
            else:
                no_decay_feat.append(param)
        else:
            if is_fc:
                decay_fc.append(param)
            else:
                decay_feat.append(param)
    return [
        {'params': no_decay_feat, 'weight_decay': 0., 'lr': feature_lr_mult * learning_rate},
        {'params': no_decay_fc, 'weight_decay': 0., 'lr': fc_lr_mult * learning_rate},
        {'params': decay_feat, 'weight_decay': weight_decay, 'lr': feature_lr_mult * learning_rate},
        {'params': decay_fc, 'weight_decay': weight_decay, 'lr': fc_lr_mult * learning_rate}]

def filter_fc_layer(model, learning_rate=0.001, fc_names=_TIMM_FC_NAMES, feature_lr_mult=0.01, fc_lr_mult=1):
    """Filter linear projection layers from the network"""
    if feature_lr_mult == 1 and fc_lr_mult == 1:
        return model.parameters()
    feat = []
    fc = []
    for name, param in model.named_parameters():
        is_fc = False
        if any([name.startswith(pattern) for pattern in fc_names]):
            is_fc = True
        if not param.requires_grad:
            continue  # frozen weights
        if is_fc:
            fc.append(param)
        else:
            feat.append(param)

    return [
        {'params': feat, 'lr': feature_lr_mult * learning_rate},
        {'params': fc, 'lr': fc_lr_mult * learning_rate}]

def create_optimizer_v2a(
        model: nn.Module,
        optimizer_name: str = 'sgd',
        learning_rate: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        filter_bias_and_bn: bool = True,
        feature_lr_mult: float = 0.01,
        fc_lr_mult: float = 1,
        **kwargs):
    """ Create an optimizer.
    Note that this version is modifed based on
    https://github.com/rwightman/pytorch-image-models/blob/cd3dc4979f6ca16a09910b4a32b7a8f07cc31fda/timm/optim/optim_factory.py#L73
    It allows feature backbone and output linear layers to have different learning rate by having more groups
    TODO currently the model is passed in and all parameters are selected for optimization.
    For more general use an interface that allows selection of parameters to optimize and lr groups, one of:
      * a filter fn interface that further breaks params into groups in a weight_decay compatible fashion
      * expose the parameters interface and leave it up to caller
    Args:
        model (nn.Module): model containing parameters to optimize
        optimizer_name: name of optimizer to create
        learning_rate: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
        filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through
    Returns:
        Optimizer
    """
    opt_lower = optimizer_name.lower()
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay_filter_fc(model, learning_rate, weight_decay, skip,
                                                _TIMM_FC_NAMES, feature_lr_mult, fc_lr_mult)
        weight_decay = 0.
    else:
        parameters = filter_fc_layer(model, learning_rate, _TIMM_FC_NAMES, feature_lr_mult, fc_lr_mult)
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=learning_rate, weight_decay=weight_decay, **kwargs)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower in ('sgd', 'nesterov'):
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adabelief':
        optimizer = AdaBelief(parameters, rectify=False, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not learning_rate:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
