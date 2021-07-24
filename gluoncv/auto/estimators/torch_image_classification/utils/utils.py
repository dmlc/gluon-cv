from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT

def update_cfg(cfg, udict):
    cfg.unfreeze()
    cfg.update(udict)
    cfg.freeze()

# pylint: disable=dangerous-default-value
def resolve_data_config(cfg, default_cfg={}, model=None, use_test_size=False):
    default_cfg = default_cfg
    if not default_cfg and model is not None and hasattr(model, 'default_cfg'):
        default_cfg = model.default_cfg

    # Resolve input/image size
    in_chans = 3
    input_size = (in_chans, 224, 224)
    if cfg.data.input_size is not None:
        assert isinstance(cfg.data.input_size, (tuple, list))
        assert len(cfg.data.input_size) == 3
        input_size = tuple(cfg.data.input_size)
        in_chans = input_size[0]  # input_size overrides in_chans
    elif cfg.data.img_size is not None:
        assert isinstance(cfg.data.img_size, int)
        input_size = (in_chans, cfg.data.img_size, cfg.data.img_size)
    else:
        if use_test_size and 'test_input_size' in default_cfg:
            input_size = default_cfg['test_input_size']
        elif 'input_size' in default_cfg:
            input_size = default_cfg['input_size']
    update_cfg(cfg, {'data': {'input_size': input_size}})

    # resolve interpolation method
    interpolation = 'bicubic'
    if cfg.data.interpolation is not None:
        interpolation = cfg.data.interpolation
    elif 'interpolation' in default_cfg:
        interpolation = default_cfg['interpolation']
    update_cfg(cfg, {'data': {'interpolation': interpolation}})

    # resolve dataset + model mean for normalization
    mean = IMAGENET_DEFAULT_MEAN
    if cfg.data.mean is not None:
        mean = tuple(cfg.data.mean)
        if len(mean) == 1:
            mean = tuple(list(mean) * in_chans)
        else:
            assert len(mean) == in_chans
    elif 'mean' in default_cfg:
        mean = default_cfg['mean']
    update_cfg(cfg, {'data': {'mean': mean}})

    # resolve dataset + model std deviation for normalization
    std = IMAGENET_DEFAULT_STD
    if cfg.data.std is not None:
        std = tuple(cfg.data.std)
        if len(std) == 1:
            std = tuple(list(std) * in_chans)
        else:
            assert len(std) == in_chans
    elif 'std' in default_cfg:
        std = default_cfg['std']
    update_cfg(cfg, {'data': {'std': std}})

    # resolve default crop percentage
    crop_pct = DEFAULT_CROP_PCT
    if cfg.data.crop_pct is not None:
        crop_pct = cfg.data.crop_pct
    elif 'crop_pct' in default_cfg:
        crop_pct = default_cfg['crop_pct']
    update_cfg(cfg, {'data': {'crop_pct': crop_pct}})
