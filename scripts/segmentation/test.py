import os
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

from gluoncv.utils import PolyLRScheduler
from gluoncv.model_zoo.segbase import *
from gluoncv.utils.parallel import *
from gluoncv.data import get_segmentation_dataset, test_batchify_fn
from gluoncv.utils.viz import get_color_pallete

from train import parse_args

def test(args):
    # output folder
    outdir = 'outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    testset = get_segmentation_dataset(
        args.dataset, split='test', transform=input_transform)
    test_data = gluon.data.DataLoader(
        testset, args.test_batch_size, last_batch='keep',
        batchify_fn=test_batchify_fn, num_workers=args.workers)
    # create network
    model = get_segmentation_model(model=args.model, dataset=args.dataset,
                                   backbone=args.backbone, norm_layer=args.norm_layer)
    print(model)
    evaluator = MultiEvalModel(model, testset.num_class, ctx_list=args.ctx)
    # load pretrained weight
    assert(args.resume is not None)
    if os.path.isfile(args.resume):
        model.load_params(args.resume, ctx=args.ctx)
    else:
        raise RuntimeError("=> no checkpoint found at '{}'" \
            .format(args.resume))

    tbar = tqdm(test_data)
    for i, (data, im_paths) in enumerate(tbar):
        predicts = evaluator.parallel_forward(data)
        for predict, impath in zip(predicts, im_paths):
            predict = mx.nd.squeeze(mx.nd.argmax(predict, 1)).asnumpy()
            mask = get_color_pallete(predict, args.dataset)
            outname = os.path.splitext(impath)[0] + '.png'
            mask.save(os.path.join(outdir, outname))


if __name__ == "__main__":
    args = parse_args()
    args.test_batch_size = args.ngpus
    print('Testing model: ', args.resume)
    test(args)
