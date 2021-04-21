import argparse, time, logging, os, math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler

# CLI
def parse_args(logger = None):
    def float_list(x):
        return list(map(float, x.split(',')))

    data_dir = '~/.mxnet/datasets/imagenet/'
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--data-backend', choices=('dali-gpu', 'dali-cpu', 'mxnet'), default='mxnet',
                        help='set data loading & augmentation backend')
    parser.add_argument('--data-dir', type=str, default=data_dir,
                        help='training and validation pictures to use.')
    parser.add_argument('--rec-train', type=str, default=data_dir+'rec/train.rec',
                        help='the training data')
    parser.add_argument('--rec-train-idx', type=str, default=data_dir+'rec/train.idx',
                        help='the index of training data')
    parser.add_argument('--rec-val', type=str, default=data_dir+'rec/val.rec',
                        help='the validation data')
    parser.add_argument('--rec-val-idx', type=str, default=data_dir+'rec/val.idx',
                        help='the index of validation data')
    parser.add_argument('--use-rec', action='store_true',
                        help='use image record iter for data input. default is false.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--last-gamma', action='store_true',
                        help='whether to init gamma of the last BN layer in each bottleneck to 0.')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--model', type=str, required=True,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='Crop ratio during validation. default is 0.875')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--use_se', action='store_true',
                        help='use SE layers or not in resnext. default is false.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether train the model with mix-up. default is false.')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='beta distribution parameter for mixup sampling, default is 0.2.')
    parser.add_argument('--mixup-off-epoch', type=int, default=0,
                        help='how many last epochs to train without mixup, default is 0.')
    parser.add_argument('--label-smoothing', action='store_true',
                        help='use label smoothing or not in training. default is false.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--teacher', type=str, default=None,
                        help='teacher model for distillation training')
    parser.add_argument('--temperature', type=float, default=20,
                        help='temperature parameter for distillation teacher model')
    parser.add_argument('--hard-weight', type=float, default=0.5,
                        help='weight for the loss of one-hot label for distillation training')
    parser.add_argument('--batch-norm', action='store_true',
                        help='enable batch normalization or not in vgg. default is false.')
    parser.add_argument('--save-frequency', type=int, default=10,
                        help='frequency of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-epoch', type=int, default=0,
                        help='epoch to resume training from.')
    parser.add_argument('--resume-params', type=str, default='',
                        help='path of parameters to load from.')
    parser.add_argument('--resume-states', type=str, default='',
                        help='path of trainer state to load from.')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Number of batches to wait before logging.')
    parser.add_argument('--logging-file', type=str, default='train_imagenet.log',
                        help='name of training log file')
    parser.add_argument('--use-gn', action='store_true',
                        help='whether to use group norm.')
    parser.add_argument('--rgb-mean', type=float_list, default=[123.68, 116.779, 103.939],
                        help='a tuple of size 3 for the mean rgb')
    parser.add_argument('--rgb-std', type=float_list, default=[58.393, 57.12, 57.375],
                        help='a tuple of size 3 for the std rgb')
    parser.add_argument('--num-training-samples', type=int, default=1281167,
                        help='Number of training samples')
    if logger:
        # DALI is expected to be used
        dali_ver = 'DALI_VER'
        dali_version = os.environ.get(dali_ver) if dali_ver in os.environ else 'nvidia-dali-cuda100'
        stream = os.popen("pip list --format=columns | grep dali")
        if stream.read() == '':
            # DALI is not installed
            cmd_install = 'pip install --extra-index-url https://developer.download.nvidia.com/compute/redist ' + dali_version
            logger.info('DALI is supposed to be used, but it is not installed.\nWe can try to install it for you (and continue this test) OR\n' \
                        'this test will be stopped and you can later restart it after installing DALI manually')
            answer = input('Do you want to install DALI now? (Y/N):')
            if answer[0] == 'Y' or answer[0] == 'y':
                logger.info('Trying to install DALI version \'%s\'' % dali_version)
                ret = os.system(cmd_install)
                if ret != 0:
                    logger.info('Cannot install DALI version \'%s\'.\n' \
                                'Perhaps, the latest DALI version should be used.\n' % dali_version)
            else:
                ret = 1
                logger.info('To install DALI, please, use:\n' + cmd_install)
            if ret != 0:
                logger.info('Please, see documentation on ' \
                            'https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html\n' \
                            'and set the environment variable %s to the appropriate version ID (default is \'%s\')' % (dali_ver, dali_version))
                raise RuntimeError('DALI is not installed')
        try:
            import dali
        except ImportError:
            raise ImportError('Unable to import modules dali.py')

        parser = dali.add_dali_args(parser)
        dali.add_data_args(parser)
    return parser.parse_args()


def main():
    def batch_func(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    opt = parse_args()

    filehandler = logging.FileHandler(opt.logging_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    if opt.data_backend[0:5] == 'dali-':
        opt = parse_args(logger)   # Adding DALI parameters

    if opt.data_backend == 'dali-gpu' and opt.num_gpus == 0:
        stream = os.popen('nvidia-smi -L | wc -l')
        opt.num_gpus = int(stream.read())
        logger.info("When '--data-backend' is equal to 'dali-gpu', then `--num-gpus` should NOT be 0\n" \
                    "For now '--num-gpus' will be set to the number of GPUs installed: %d" % opt.num_gpus)

    logger.info(opt)

    classes = 1000
    num_gpus = opt.num_gpus
    batch_size = opt.batch_size * max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    opt.gpus = [i for i in range(num_gpus)]

    lr_decay = opt.lr_decay
    lr_decay_period = opt.lr_decay_period
    if opt.lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
    lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]
    num_batches = opt.num_training_samples // batch_size

    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=opt.lr,
                    nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
                    nepochs=opt.num_epochs - opt.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=lr_decay, power=2)
    ])

    model_name = opt.model

    kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'classes': classes}
    if opt.use_gn:
        kwargs['norm_layer'] = gcv.nn.GroupNorm
    if model_name.startswith('vgg'):
        kwargs['batch_norm'] = opt.batch_norm
    elif model_name.startswith('resnext'):
        kwargs['use_se'] = opt.use_se

    if opt.last_gamma:
        kwargs['last_gamma'] = True

    optimizer = 'nag'
    optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum, 'lr_scheduler': lr_scheduler}
    if opt.dtype != 'float32':
        optimizer_params['multi_precision'] = True

    net = get_model(model_name, **kwargs)
    net.cast(opt.dtype)
    if opt.resume_params != '':
        net.load_parameters(opt.resume_params, ctx = context)

    # teacher model for distillation training
    distillation = opt.teacher is not None and opt.hard_weight < 1.0
    if distillation:
        teacher = get_model(opt.teacher, pretrained=True, classes=classes, ctx=context)
        teacher.cast(opt.dtype)

    # Two functions for reading data from record file or raw images
    def get_data_rec(args):
        rec_train = os.path.expanduser(args.rec_train)
        rec_train_idx = os.path.expanduser(args.rec_train_idx)
        rec_val = os.path.expanduser(args.rec_val)
        rec_val_idx = os.path.expanduser(args.rec_val_idx)
        num_gpus = args.num_gpus
        batch_size = args.batch_size * max(1, num_gpus)

        jitter_param = 0.4
        lighting_param = 0.1
        input_size = opt.input_size
        crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
        resize = int(math.ceil(input_size / crop_ratio))

        train_data = mx.io.ImageRecordIter(
            path_imgrec         = rec_train,
            path_imgidx         = rec_train_idx,
            preprocess_threads  = args.num_workers,
            shuffle             = True,
            batch_size          = batch_size,
            data_shape          = (3, input_size, input_size),
            mean_r              = args.rgb_mean[0],
            mean_g              = args.rgb_mean[1],
            mean_b              = args.rgb_mean[2],
            std_r               = args.rgb_std[0],
            std_g               = args.rgb_std[1],
            std_b               = args.rgb_std[2],
            rand_mirror         = True,
            random_resized_crop = True,
            max_aspect_ratio    = 4. / 3.,
            min_aspect_ratio    = 3. / 4.,
            max_random_area     = 1,
            min_random_area     = 0.08,
            brightness          = jitter_param,
            saturation          = jitter_param,
            contrast            = jitter_param,
            pca_noise           = lighting_param,
        )
        val_data = mx.io.ImageRecordIter(
            path_imgrec         = rec_val,
            path_imgidx         = rec_val_idx,
            preprocess_threads  = args.num_workers,
            shuffle             = False,
            batch_size          = batch_size,

            resize              = resize,
            data_shape          = (3, input_size, input_size),
            mean_r              = args.rgb_mean[0],
            mean_g              = args.rgb_mean[1],
            mean_b              = args.rgb_mean[2],
            std_r               = args.rgb_std[0],
            std_g               = args.rgb_std[1],
            std_b               = args.rgb_std[2],
        )
        return train_data, val_data, batch_func

    def get_data_rec_transfomed(args):
        data_dir = args.data_dir
        num_workers = args.num_workers
        batch_size = args.batch_size * max(1, args.num_gpus)
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        jitter_param = 0.4
        lighting_param = 0.1
        input_size = opt.input_size
        crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
        resize = int(math.ceil(input_size / crop_ratio))

        def batch_fn(batch, ctx):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            return data, label

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomFlipLeftRight(),
            transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                        saturation=jitter_param),
            transforms.RandomLighting(lighting_param),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize(resize, keep_ratio=True),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize
        ])

        train_data = gluon.data.DataLoader(
            imagenet.classification.ImageNet(data_dir, train=True).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
        val_data = gluon.data.DataLoader(
            imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_data, val_data, batch_fn

    def get_data_loader(args):
        if args.data_backend == 'mxnet':
            return get_data_rec if args.use_rec else get_data_rec_transfomed

        # Check if DALI is installed:
        if args.data_backend[0:5] == 'dali-':
            import dali
        if args.data_backend == 'dali-gpu':
            return (lambda *args, **kwargs: dali.get_rec_iter(*args, **kwargs, batch_fn=batch_func, dali_cpu=False))
        if args.data_backend == 'dali-cpu':
            return (lambda *args, **kwargs: dali.get_rec_iter(*args, **kwargs, batch_fn=batch_func, dali_cpu=True))
        raise ValueError('Wrong data backend')


    train_data, val_data, batch_fn = get_data_loader(opt)(opt)
    train_metric = mx.metric.RMSE() if opt.mixup else mx.metric.Accuracy()
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    save_frequency = opt.save_frequency
    if opt.save_dir and save_frequency:
        save_dir = opt.save_dir
        makedirs(save_dir)
    else:
        save_dir = ''
        save_frequency = 0

    def mixup_transform(label, classes, lam=1, eta=0.0):
        if isinstance(label, nd.NDArray):
            label = [label]
        res = []
        for l in label:
            y1 = l.one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
            y2 = l[::-1].one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
            res.append(lam*y1 + (1-lam)*y2)
        return res

    def smooth(label, classes, eta=0.1):
        if isinstance(label, nd.NDArray):
            label = [label]
        smoothed = []
        for l in label:
            res = l.one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
            smoothed.append(res)
        return smoothed

    def test(ctx, val_data, val_batch):
        if opt.use_rec:
            val_data.reset()
        acc_top1.reset()
        acc_top5.reset()
        for i, batch in enumerate(val_data):
            for j in range(val_batch):
                data, label = batch_fn(batch[j], ctx) if type(batch) == list else batch_fn(batch, ctx)
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
                acc_top1.update(label, outputs)
                acc_top5.update(label, outputs)

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        return (1-top1, 1-top5)

    def train(ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        if opt.resume_params == '':
            net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

        if opt.no_wd:
            for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
        if opt.resume_states != '':
            trainer.load_states(opt.resume_states)

        sparse_label_loss = not (opt.label_smoothing or opt.mixup)
        if distillation:
            L = gcv.loss.DistillationSoftmaxCrossEntropyLoss(temperature=opt.temperature,
                                                                 hard_weight=opt.hard_weight,
                                                                 sparse_label=sparse_label_loss)
        else:
            L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label_loss)

        best_val_score = 1
        eta = 0.1 if opt.label_smoothing else 0.0
        start_time = time.time()
        val_batch = len(ctx) if opt.data_backend != 'mxnet' else 1
        for epoch in range(opt.resume_epoch, opt.num_epochs):
            tic = time.time()
            if opt.use_rec:
                train_data.reset()
            train_metric.reset()
            btic = time.time()

            for i, batch in enumerate(train_data):
                for j in range(val_batch):
                    data, label = batch_fn(batch[j], ctx) if type(batch) == list else batch_fn(batch, ctx)
                    if opt.mixup:
                        lam = np.random.beta(opt.mixup_alpha, opt.mixup_alpha)
                        if epoch >= opt.num_epochs - opt.mixup_off_epoch:
                            lam = 1
                        data = [lam*X + (1-lam)*X[::-1] for X in data]
                        label = mixup_transform(label, classes, lam, eta)

                    elif opt.label_smoothing:
                        hard_label = label
                        label = smooth(label, classes)

                    if distillation:
                        teacher_prob = [nd.softmax(teacher(X.astype(opt.dtype, copy=False)) / opt.temperature) \
                                        for X in data]

                    with ag.record():
                        outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
                        if distillation:
                            loss = [L(yhat.astype('float32', copy=False),
                                      y.astype('float32', copy=False),
                                      p.astype('float32', copy=False)) for yhat, y, p in zip(outputs, label, teacher_prob)]
                        else:
                            loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]
                    for l in loss:
                        l.backward()
                    trainer.step(batch_size)

                    if opt.mixup:
                        output_softmax = [nd.SoftmaxActivation(out.astype('float32', copy=False)) for out in outputs]
                        train_metric.update(label, output_softmax)
                    else:
                        if opt.label_smoothing:
                            train_metric.update(hard_label, outputs)
                        else:
                            train_metric.update(label, outputs)

                if opt.log_interval and not (i+1)%opt.log_interval:
                    train_metric_name, train_metric_score = train_metric.get()
                    logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f'%(
                                epoch, i+1, batch_size*opt.log_interval*val_batch/(time.time()-btic),
                                train_metric_name, train_metric_score, trainer.learning_rate))
                    btic = time.time()

            train_metric_name, train_metric_score = train_metric.get()
            if opt.log_interval and i % opt.log_interval:
                # We did NOT report the speed on the last iteration of the loop. Let's do it now
                logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f'%(
                            epoch, i, batch_size*(i%opt.log_interval)*val_batch/(time.time()-btic),
                            train_metric_name, train_metric_score, trainer.learning_rate))

            epoch_time = time.time() - tic
            throughput = int(batch_size * i * val_batch / epoch_time)
            err_top1_val, err_top5_val = test(ctx, val_data, val_batch)

            logger.info('[Epoch %d] training: %s=%f'%(epoch, train_metric_name, train_metric_score))
            logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f'%(epoch, throughput, epoch_time))
            logger.info('[Epoch %d] validation: err-top1=%f err-top5=%f'%(epoch, err_top1_val, err_top5_val))

            if err_top1_val < best_val_score:
                best_val_score = err_top1_val
                net.save_parameters('%s/%.4f-imagenet-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))
                trainer.save_states('%s/%.4f-imagenet-%s-%d-best.states'%(save_dir, best_val_score, model_name, epoch))

            if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
                net.save_parameters('%s/imagenet-%s-%d.params'%(save_dir, model_name, epoch))
                trainer.save_states('%s/imagenet-%s-%d.states'%(save_dir, model_name, epoch))

        if save_frequency and save_dir:
            net.save_parameters('%s/imagenet-%s-%d.params'%(save_dir, model_name, opt.num_epochs-1))
            trainer.save_states('%s/imagenet-%s-%d.states'%(save_dir, model_name, opt.num_epochs-1))

        logger.info('Training time for %d epochs: %f sec.'%(opt.num_epochs, time.time() - start_time))

    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
        if distillation:
            teacher.hybridize(static_alloc=True, static_shape=True)
    train(context)

if __name__ == '__main__':
    main()
