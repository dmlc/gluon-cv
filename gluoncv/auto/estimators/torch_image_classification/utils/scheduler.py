from timm.scheduler import CosineLRScheduler, PlateauLRScheduler,\
                           StepLRScheduler, TanhLRScheduler

def create_scheduler(cfg, optimizer):
    num_epochs = cfg.train.epochs

    if cfg.train.lr_noise is not None:
        lr_noise = cfg.train.lr_noise
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None

    lr_scheduler = None
    if cfg.train.sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=getattr(cfg.train, 'lr_cycle_mul', 1.),
            lr_min=cfg.train.min_lr,
            decay_rate=cfg.train.decay_rate,
            warmup_lr_init=cfg.train.warmup_lr,
            warmup_t=cfg.train.warmup_epochs,
            cycle_limit=getattr(cfg.train, 'lr_cycle_limit', 1),
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=getattr(cfg.train, 'lr_noise_pct', 0.67),
            noise_std=getattr(cfg.train, 'lr_noise_std', 1.),
            noise_seed=getattr(cfg.misc, 'seed', 42),
        )
        num_epochs = lr_scheduler.get_cycle_length() + cfg.train.cooldown_epochs
    elif cfg.train.sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=getattr(cfg.train, 'lr_cycle_mul', 1.),
            lr_min=cfg.train.min_lr,
            warmup_lr_init=cfg.train.warmup_lr,
            warmup_t=cfg.train.warmup_epochs,
            cycle_limit=getattr(cfg.train, 'lr_cycle_limit', 1),
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=getattr(cfg.train, 'lr_noise_pct', 0.67),
            noise_std=getattr(cfg.train, 'lr_noise_std', 1.),
            noise_seed=getattr(cfg.misc, 'seed', 42),
        )
        num_epochs = lr_scheduler.get_cycle_length() + cfg.train.cooldown_epochs
    elif cfg.train.sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=cfg.train.decay_epochs,
            decay_rate=cfg.train.decay_rate,
            warmup_lr_init=cfg.train.warmup_lr,
            warmup_t=cfg.train.warmup_epochs,
            noise_range_t=noise_range,
            noise_pct=getattr(cfg.train, 'lr_noise_pct', 0.67),
            noise_std=getattr(cfg.train, 'lr_noise_std', 1.),
            noise_seed=getattr(cfg.misc, 'seed', 42),
        )
    elif cfg.train.sched == 'plateau':
        mode = 'min' if 'loss' in getattr(cfg.misc, 'eval_metric', '') else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=cfg.train.decay_rate,
            patience_t=cfg.train.patience_epochs,
            lr_min=cfg.train.min_lr,
            mode=mode,
            warmup_lr_init=cfg.train.warmup_lr,
            warmup_t=cfg.train.warmup_epochs,
            cooldown_t=0,
            noise_range_t=noise_range,
            noise_pct=getattr(cfg.train, 'lr_noise_pct', 0.67),
            noise_std=getattr(cfg.train, 'lr_noise_std', 1.),
            noise_seed=getattr(cfg.misc, 'seed', 42),
        )

    return lr_scheduler, num_epochs
