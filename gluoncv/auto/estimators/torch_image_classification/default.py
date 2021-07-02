"""Default configs for torch image classification"""
# pylint: disable=bad-whitespace,missing-class-docstring
from typing import Union, Tuple, List
from autocfg import dataclass, field

@dataclass
class ModelCfg:
    model: str = 'resnet101'
    pretrained: bool = False
    initial_checkpoint: str = ''  # Initialize model from this checkpoint
    resume: str = ''  # Resume full model and optimizer state from checkpoint
    no_resume_opt: bool = False  # prevent resume of optimizer state when resuming model
    global_pool_type: Union(str, None) = None  # Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None
    

@dataclass
class DatasetCfg:
    data_dir: str = ''  # path to dataset
    dataset: str = ''  # dataset type (default: ImageFolder/ImageTar if empty)
    train_split: str = 'train'  # dataset train split
    val_split: str = 'validation'  # dataset validation split
    img_size: Union(int, None) = None  # Image patch size (default: None => model default)
    input_size: Union(Tuple[int, int, int], None) = None  # Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty
    crop_pct: Union(float, None) = None  # Input image center crop percent (for validation only)
    mean: Union(Tuple[float, ...], None) = None  # Override mean pixel value of dataset
    std : Union(Tuple[float, ...], None) = None  # Override std deviation of of dataset
    interpolation: str = ''  # Image resize interpolation type (overrides model)
    batch_size: int = 32
    validation_batch_size_multiplier: int = 1  # ratio of validation batch size to training batch size (default: 1)

@dataclass
class OptimizerCfg:
    opt: str = 'sgd'
    opt_eps: Union(float, None) = None  # Optimizer Epsilon (default: None, use opt default)
    opt_betas: Union(Tuple[float, ...], None) = None  # Optimizer Betas (default: None, use opt default)
    momentum: float = 0.9
    weight_decay: float = 0.0001
    clip_grad: Union(float, None) = None  # Clip gradient norm (default: None, no clipping)
    clip_mode: str = 'norm'  # Gradient clipping mode. One of ("norm", "value", "agc")

@dataclass
class TrainCfg:
    sched: str = 'step'  # LR scheduler
    lr: float = 0.01
    lr_noise: Union(Tuple[float, ...], None) = None  # learning rate noise on/off epoch percentages
    lr_noise_pct: float = 0.67  # learning rate noise limit percent
    lr_noise_std: float = 1.0  # learning rate noise std-dev
    lr_cycle_mul: float = 1.0  # learning rate cycle len multiplier
    lr_cycle_limit: int = 1 # learning rate cycle limit
    warmup_lr: float = 0.0001
    min_lr: float = 1e-5
    epochs: int = 200
    epoch_repeats: float = 0.  # epoch repeat multiplier (number of times to repeat dataset epoch per train epoch)
    start_epoch: Union(int, None) = None  # manual epoch number (useful on restarts)
    decay_epochs: float = 30  # epoch interval to decay LR
    warmup_epochs: int = 3  # epochs to warmup LR, if scheduler supports
    cooldown_epochs: int = 10  # epochs to cooldown LR at min_lr, after cyclic schedule ends
    patience_epochs: int = 10  # patience epochs for Plateau LR scheduler
    decay_rate: float = 0.1
    bn_momentum: Union[float, None] = None  # BatchNorm momentum override
    bn_eps: Union[float, None] = None  # BatchNorm epsilon override
    sync_bn: bool = False  # Enable NVIDIA Apex or Torch synchronized BatchNorm
    dist_bn: str = ''  # Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")
    split_bn: bool = False  # Enable separate BN layers per augmentation split

@dataclass
class AugmentationCfg:
    no_aug: bool = False  # Disable all training augmentation, override other train aug args
    scale: List[float] = [0.08, 1.0]  # Random resize scale
    ratio: List[float] = [3./4., 4./3.]  # Random resize aspect ratio (default: 0.75 1.33
    hflip: float = 0.5  # Horizontal flip training aug probability
    vflip: float = 0.5 # Vertical flip training aug probability
    color_jitter: float = 0.4
    aa: Union[str, None] = None  # Use AutoAugment policy. "v0" or "original
    aug_splits: int = 0  # Number of augmentation splits (default: 0, valid: 0 or >=2)
    jsd: bool = False  # 'Enable Jensen-Shannon Divergence + CE loss. Use with `aug_splits`
    reprob: float = 0  # Random erase prob
    remode: str = 'const'  # Random erase mode
    recount: int = 1  # Random erase count
    resplit: bool = False  # Do not random erase first (clean) augmentation split
    mixup: float = 0.0  # mixup alpha, mixup enabled if > 0
    cut_mix: float = 0.0  # cutmix alpha, cutmix enabled if > 0
    cutmix_minmax: Union[Tuple[float, ...], None] = None  # cutmix min/max ratio, overrides alpha and enables cutmix if set
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
    model_ema: bool = False  # Enable tracking moving average of model weights
    model_ema_force_cpu: bool = False # Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation
    model_ema_decay: float = 0.9998  # decay factor for model weights moving average

@dataclass
class MiscCfg:
    seed: int = 42
    log_interval: int = 50  # how many batches to wait before logging training status
    recovery_interval: int = 0  # how many batches to wait before writing recovery checkpoint
    checkpoint_hist: int = 10  # number of checkpoints to keep
    num_workers: int = 4  # how many training processes to use
    save_images: bool = False  # save images of input bathes every log interval for debugging
    amp: bool = False  # use NVIDIA Apex AMP or Native AMP for mixed precision training
    apex_amp: bool = False,  # Use NVIDIA Apex AMP mixed precision
    native_amp: bool = False  # Use Native Torch AMP mixed precision
    channels_last: bool = False  # Use channels_last memory layout
    pin_mem: bool = False  # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU
    no_prefetcher: bool = False  # disable fast prefetcher
    output: Union[str, None] = None  # path to output folder (default: none, current dir)
    eval_metric: str = 'top1'  # 'Best metric (default: "top1")
    tta: int = 0  # Test/inference time augmentation (oversampling) factor. 0=None
    local_rank: int = 0 
    use_multi_epochs_loader: bool = False  # use the multi-epochs-loader to save time at the beginning of every epoch
    torchscript: bool = False  #convert model torchscript for inference

@dataclass
class TorchImageClassificationCfg:
    model : ModelCfg = field(default_factory=ModelCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    augmentation: AugmentationCfg = field(default_factory=AugmentationCfg)
    model_ema: ModelEMACfg = field(default_factory=ModelEMACfg)
    misc: MiscCfg = field(default_factory=MiscCfg)
    gpus : Union[Tuple, list] = (0, )  # gpu individual ids, not necessarily consecutive
