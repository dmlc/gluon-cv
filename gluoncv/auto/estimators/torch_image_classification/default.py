"""Default configs for torch image classification"""
# pylint: disable=bad-whitespace,missing-class-docstring
from typing import Union, Tuple
from autocfg import dataclass, field

@dataclass
class ImageClassification:
    model: str = 'resnet101'
    pretrained: bool = True
    global_pool_type: Union[str, None] = None  # Global pool type, one of (fast, avg, max, avgmax). Model default if None

@dataclass
class DataCfg:
    img_size: Union[int, None] = None  # Image patch size (default: None => model default)
    input_size: Union[Tuple[int, int, int], None] = None  # Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty
    crop_pct: Union[float, None] = 0.99  # Input image center crop percent (for validation only)
    mean: Union[Tuple, None] = None  # Override mean pixel value of dataset
    std : Union[Tuple, None] = None  # Override std deviation of of dataset
    interpolation: str = ''  # Image resize interpolation type (overrides model)
    validation_batch_size_multiplier: int = 1  # ratio of validation batch size to training batch size (default: 1)

@dataclass
class OptimizerCfg:
    opt: str = 'sgd'
    opt_eps: Union[float, None] = None  # Optimizer Epsilon (default: None, use opt default)
    opt_betas: Union[Tuple, None] = None  # Optimizer Betas (default: None, use opt default)
    momentum: float = 0.9
    weight_decay: float = 0.0001
    clip_grad: Union[float, None] = None  # Clip gradient norm (default: None, no clipping)
    clip_mode: str = 'norm'  # Gradient clipping mode. One of ("norm", "value", "agc")

@dataclass
class TrainCfg:
    batch_size: int = 32
    sched: str = 'step'  # LR scheduler
    lr: float = 0.01
    lr_noise: Union[Tuple, None] = None  # learning rate noise on/off epoch percentages
    lr_noise_pct: float = 0.67  # learning rate noise limit percent
    lr_noise_std: float = 1.0  # learning rate noise std-dev
    lr_cycle_mul: float = 1.0  # learning rate cycle len multiplier
    lr_cycle_limit: int = 1 # learning rate cycle limit
    transfer_lr_mult : float = 0.01  # reduce the backbone lr_mult to avoid quickly destroying the features
    output_lr_mult : float = 0.1  # the learning rate multiplier for last fc layer if trained with transfer learning
    warmup_lr: float = 0.0001
    min_lr: float = 1e-5
    epochs: int = 200
    start_epoch: int = 0  # manual epoch number (useful on restarts)
    decay_epochs: int = 30  # epoch interval to decay LR
    warmup_epochs: int = 3  # epochs to warmup LR, if scheduler supports
    cooldown_epochs: int = 10  # epochs to cooldown LR at min_lr, after cyclic schedule ends
    patience_epochs: int = 10  # patience epochs for Plateau LR scheduler
    decay_rate: float = 0.1
    bn_momentum: Union[float, None] = None  # BatchNorm momentum override
    bn_eps: Union[float, None] = None  # BatchNorm epsilon override
    sync_bn: bool = False  # Enable NVIDIA Apex or Torch synchronized BatchNorm
    early_stop_patience : int = -1  # epochs with no improvement after which train is early stopped, negative: disabled
    early_stop_min_delta : float = 0.001  # ignore changes less than min_delta for metrics
    # the baseline value for metric, training won't stop if not reaching baseline
    early_stop_baseline : Union[float, int] = 0.0
    early_stop_max_value : Union[float, int] = 1.0  # early stop if reaching max value instantly

@dataclass
class AugmentationCfg:
    no_aug: bool = False  # Disable all training augmentation, override other train aug args
    scale: Tuple[float, float] = (0.08, 1.0)  # Random resize scale
    ratio: Tuple[float, float] = (3./4., 4./3.) # Random resize aspect ratio (default: 0.75 1.33
    hflip: float = 0.5  # Horizontal flip training aug probability
    vflip: float = 0.0 # Vertical flip training aug probability
    color_jitter: float = 0.4
    auto_augment: Union[str, None] = None  # Use AutoAugment policy. "v0" or "original
    mixup: float = 0.0  # mixup alpha, mixup enabled if > 0
    cutmix: float = 0.0  # cutmix alpha, cutmix enabled if > 0
    cutmix_minmax: Union[Tuple, None] = None  # cutmix min/max ratio, overrides alpha and enables cutmix if set
    mixup_prob: float = 1.0  # Probability of performing mixup or cutmix when either/both is enabled
    mixup_switch_prob: float = 0.5  # Probability of switching to cutmix when both mixup and cutmix enabled
    mixup_mode: str = 'batch'  # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
    mixup_off_epoch: int = 0  # Turn off mixup after this epoch, disabled if 0
    smoothing: float = 0.1  # Label smoothin
    train_interpolation: str = 'random'  # Training interpolation (random, bilinear, bicubic)
    drop: float = 0.0  # Dropout rate
    drop_path: Union[float, None] = None  # Drop path rate
    drop_block: Union[float, None] = None # Drop block rate

@dataclass
class ModelEMACfg:
    model_ema: bool = True  # Enable tracking moving average of model weights
    model_ema_force_cpu: bool = False # Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation
    model_ema_decay: float = 0.9998  # decay factor for model weights moving average

@dataclass
class MiscCfg:
    seed: int = 42
    log_interval: int = 50  # how many batches to wait before logging training status
    num_workers: int = 4  # how many training processes to use
    save_images: bool = False  # save images of input bathes every log interval for debugging
    amp: bool = False  # use NVIDIA Apex AMP or Native AMP for mixed precision training
    apex_amp: bool = False  # Use NVIDIA Apex AMP mixed precision
    native_amp: bool = False  # Use Native Torch AMP mixed precision
    pin_mem: bool = False  # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU
    prefetcher: bool = False  # use fast prefetcher
    eval_metric: str = 'top1'  # 'Best metric (default: "top1")
    tta: int = 0  # Test/inference time augmentation (oversampling) factor. 0=None
    use_multi_epochs_loader: bool = False  # use the multi-epochs-loader to save time at the beginning of every epoch
    torchscript: bool = False  # keep false, convert model torchscript for inference

@dataclass
class TorchImageClassificationCfg:
    img_cls : ImageClassification = field(default_factory=ImageClassification)
    data: DataCfg = field(default_factory=DataCfg)
    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    augmentation: AugmentationCfg = field(default_factory=AugmentationCfg)
    model_ema: ModelEMACfg = field(default_factory=ModelEMACfg)
    misc: MiscCfg = field(default_factory=MiscCfg)
    gpus : Union[Tuple, list] = (0, )  # gpu individual ids, not necessarily consecutive
