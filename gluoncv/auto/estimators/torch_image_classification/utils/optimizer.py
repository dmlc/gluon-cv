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
    if cfg.optimizer.opt_args is not None:
        kwargs.update(cfg.optimizer.opt_args)
    return kwargs
