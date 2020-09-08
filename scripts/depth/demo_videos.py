import os
import PIL.Image as pil
import numpy as np

import mxnet as mx
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo.monodepthv2.layers import disp_to_depth


# using cpu
ctx = mx.cpu(0)


def reading_img(root, height=192, width=640):
    files = os.listdir(root)
    files.sort()

    raw_img_squences = []
    for file in files:
        file = os.path.join(root, file)
        img = pil.open(file).convert('RGB')
        raw_img_squences.append(img)

    original_width, original_height = raw_img_squences[0].size

    return raw_img_squences, original_width, original_height


if __name__ == '__main__':
    ############################ Loading Data ############################
    # 1. loading image squences or videos
    # 2. saving it to list

    root = '2011_09_28/2011_09_28_drive_0001_sync/image_02/data'
    feed_height = 192
    feed_width = 640
    raw_img_squences, original_width, original_height = \
        reading_img(root=root, height=feed_height, width=feed_width)

    ############################ Prepare Models and Prediction ############################
    # 1. loading pretrained model
    # 2. inference
    # 3. disp to depth

    min_depth = 0.1
    max_depth = 100
    scale_factor = 5.4

    model = gluoncv.model_zoo.get_model('monodepth2_resnet18_kitti_mono_stereo_640x192',
                                        pretrained_base=False, ctx=ctx, pretrained=True)
    pred_squences = []
    for img in raw_img_squences:
        img = img.resize((feed_width, feed_height), pil.LANCZOS)
        img = transforms.ToTensor()(mx.nd.array(img)).expand_dims(0).as_in_context(context=ctx)

        outputs = model.predict(img)
        disp = outputs[("disp", 0)]
        disp, _ = disp_to_depth(disp, min_depth, max_depth)
        disp_resized = mx.nd.contrib.BilinearResize2D(disp, height=original_height, width=original_width)

        depth_pred = disp_resized * scale_factor
        depth_pred = np.clip(depth_pred, a_min=1e-3, a_max=80)

        pred_squences.append(depth_pred)

    ############################ Visualization & Store Videos ############################
    # 1. visualization, using cv2
    # 2. concat prediction and inputs
    # 2. store it as videos
    import matplotlib as mpl
    import matplotlib.cm as cm

    # from gluoncv.utils import try_import_cv2
    # cv2 = try_import_cv2()
    import cv2

    output_squences = []
    for raw_img, pred in zip(raw_img_squences, pred_squences):
        disp_resized_np = pred.squeeze().as_in_context(mx.cpu()).asnumpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        # TODO: save prediction to pred dir

        raw_img = np.array(raw_img)
        pred = np.array(im)
        output = np.concatenate((raw_img, pred), axis=0)
        output_squences.append(output)

    for output in output_squences:
        cv2.imshow('demo', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        cv2.waitKey(100)

    # TODO: save videos

    cv2.waitKey(0)
    cv2.destroyAllWindows()
