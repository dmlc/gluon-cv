from sacred import Experiment, Ingredient

faster_rcnn = Ingredient('faster_rcnn')
train_hp = Ingredient('train_hp')
valid_hp = Ingredient('valid_hp')


@faster_rcnn.config
def faster_rcnn_default():
    network = 'resnet50_v1b'  # base feature network
    dataset = 'voc'  # dataset
    nms_thresh = 0.5
    nms_topk = -1
    post_nms = -1
    roi_mode = 'align'
    roi_size = (7, 7)
    strides = (4, 8, 16, 32, 64)
    clip = 4.14
    rpn_channel = 256
    anchor_base_size = 16
    anchor_aspect_ratio = (0.5, 1, 2)
    anchor_scales = (2, 4, 8, 16, 32)
    anchor_alloc_size = (384, 384)
    rpn_nms_thresh = 0.7
    max_num_gt = 100
    gpus = (0, 1, 2, 3, 4, 5, 6, 7)
    norm_layer = None
    use_fpn = True
    custom_model = True
    num_fpn_filters = 256
    num_box_head_conv = 4
    num_box_head_conv_filters = 256
    num_box_head_dense_filters = 1024
    image_short = 800
    image_max_size = 1333
    amp = False
    static_alloc = False


@train_hp.config
def train_cfg():
    pretrained_base = True  # whether load the imagenet pre-trained base
    batch_size = 16
    start_epoch = 0
    epochs = 26
    lr = 0.01  # learning rate
    lr_decay = 0.1  # decay rate of learning rate.
    lr_decay_epoch = (20, 24)  # epochs at which learning rate decays
    lr_mode = 'step'  # learning rate scheduler mode. options are step, poly and cosine
    lr_warmup = 500  # number of iterations for warmup.
    lr_warmup_factor = 1. / 3.  # starging lr warmup factor.
    momentum = 0.9  # momentum
    wd = 1e-4  # weight decay
    log_interval = 100  # log interval
    seed = 233
    verbose = False
    mixup = False
    no_mixup_epochs = 20
    rpn_smoothl1_rho = 0.001
    rcnn_smoothl1_rho = 0.001
    horovod = False
    no_pretrained_base = False
    rpn_train_pre_nms = 12000
    rpn_train_post_nms = 2000
    rpn_min_size = 1
    rcnn_num_samples = 512
    rcnn_pos_iou_thresh = 0.5
    rcnn_pos_ratio = 0.25
    executor_threads = 4


@valid_hp.config
def valid_cfg():
    rpn_test_pre_nms = 6000
    rpn_test_post_nms = 1000
    val_interval = 1  # Epoch interval for validation


ex = Experiment('faster_rcnn_default', ingredients=[train_hp, valid_hp, faster_rcnn])


@ex.config
def default_configs():
    dataset = 'coco'
    resume = ''
    save_prefix = ''
    save_interval = 1  # save interval in epoch
    horovod = False
    num_workers = 16
    kv_store = 'nccl'
    disable_hybridization = False

