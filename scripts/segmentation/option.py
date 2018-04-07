import argparse
import os
import mxnet as mx

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='MXNet Gluon \
                                         Segmentation')
        # model and dataset 
        parser.add_argument('--model', type=str, default='fcn',
                            help='model name (default: fcn)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--dataset', type=str, default='pascalaug',
                            help='dataset name (default: pascal)')
        parser.add_argument('--nclass', type=int, default=None,
                            help='nclass for pre-trained model (default: None)')
        parser.add_argument('--workers', type=int, default=16,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--data-folder', type=str,
                            default=os.path.expanduser('~/.mxnet/datasets/voc'),
                            help='training dataset folder (default: \
                            $(HOME)/data/)')
        # training hyper params
        parser.add_argument('--aux', action='store_true', default= False,
                            help='Auxilary loss')
        parser.add_argument('--epochs', type=int, default=50, metavar='N',
                            help='number of epochs to train (default: 50)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                            training (default: 16)')
        parser.add_argument('--test-batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                            testing (default: 32)')
        parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                            help='learning rate (default: 1e-3)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        parser.add_argument('--kvstore', type=str, default='device',
                            help='kvstore to use for trainer/module.')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=
                            False, help='disables CUDA training')
        parser.add_argument('--ngpus', type=int,
                            default=len(mx.test_utils.list_gpus()),
                            help='number of GPUs (default: 4)')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default= False,
                            help='evaluating mIoU')
        # test option
        parser.add_argument('--test', action='store_true', default= False,
                            help='test a set of images and save the \
                            prediction')
        # synchronized Batch Normalization
        parser.add_argument('--syncbn', action='store_true', default= False,
                            help='using Synchronized Cross-GPU BatchNorm')
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # CUDA and GPUs
        args.cuda = not args.no_cuda
        if args.cuda:
            print('Number of GPUs:', args.ngpus)
            args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
        else:
            print('Using CPU')
            self.kvstore = 'local'
            args.ctx = mx.cpu(0)
        # model Default mean and std
        args.mean = [.485, .456, .406]
        args.std = [.229, .224, .225]
        if (args.eval or args.test) and args.resume is None:
            raise RuntimeError('checkpoint must be provided for eval or test')
        # settings for the datasets
        args.bg = False
        args.ignore_index = None
        if args.dataset == 'pascal_voc' or args.dataset == 'pascal_aug' or \
                args.dataset == 'coco':
            args.nclass = 22
            # ignoring background
            args.ignore_index = 21
        elif args.dataset == 'ade20k':
            args.nclass, args.bg = 151, True
        elif args.dataset == 'folder':
            assert (args.test and args.nclass is not None), \
                'Testing images in a folder requires --nclass for pre-trained model'
        else:
            if args.nclass is None:
                raise RuntimeError ('Customized dataset %s must provide --nclass' +
                    'during the training.'%(args.dataset))
        
        if args.syncbn:
            from gluonvision.model_zoo.syncbn import BatchNorm
            args.norm_layer = BatchNorm
        else:
            args.norm_layer = mx.gluon.nn.BatchNorm
        return args
