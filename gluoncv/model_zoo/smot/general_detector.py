import mxnet as mx
import logging
from .ssd.ssd import get_ssd
from .utils import mxnet_frame_preprocessing, timeit_context
from .utils import remap_bboxes as _remap_bboxes
from gluoncv.data import COCODetection
from gluoncv.model_zoo.smot.presets import *
import numpy as np

ssd_base_models={'ssd_300_vgg16_atrous_voc': ssd_300_vgg16_atrous_voc,
    'ssd_300_vgg16_atrous_coco': ssd_300_vgg16_atrous_coco,
    'ssd_300_vgg16_atrous_custom': ssd_300_vgg16_atrous_custom,
    'ssd_512_vgg16_atrous_voc': ssd_512_vgg16_atrous_voc,
    'ssd_512_vgg16_atrous_coco': ssd_512_vgg16_atrous_coco,
    'ssd_512_vgg16_atrous_custom': ssd_512_vgg16_atrous_custom,
    'ssd_512_resnet18_v1_voc': ssd_512_resnet18_v1_voc,
    'ssd_512_resnet18_v1_coco': ssd_512_resnet18_v1_coco,
    'ssd_512_resnet50_v1_voc': ssd_512_resnet50_v1_voc,
    'ssd_512_resnet50_v1_coco': ssd_512_resnet50_v1_coco,
    'ssd_512_resnet50_v1_custom': ssd_512_resnet50_v1_custom,
    'ssd_512_resnet101_v2_voc': ssd_512_resnet101_v2_voc,
    'ssd_512_resnet152_v2_voc': ssd_512_resnet152_v2_voc,
    'ssd_512_mobilenet1.0_voc': ssd_512_mobilenet1_0_voc,
    'ssd_512_mobilenet1.0_coco': ssd_512_mobilenet1_0_coco,
    'ssd_300_mobilenet1.0_lite_coco': ssd_300_mobilenet1_0_lite_coco,
    'ssd_512_mobilenet1.0_custom': ssd_512_mobilenet1_0_custom,
    'ssd_300_mobilenet0.25_voc': ssd_300_mobilenet0_25_voc,
    'ssd_300_mobilenet0.25_coco': ssd_300_mobilenet0_25_coco,
    'ssd_300_mobilenet0.25_custom': ssd_300_mobilenet0_25_custom,
    'ssd_300_resnet34_v1b_voc': ssd_300_resnet34_v1b_voc,
    'ssd_300_resnet34_v1b_coco': ssd_300_resnet34_v1b_coco,
    'ssd_300_resnet34_v1b_custom': ssd_300_resnet34_v1b_custom,}

def get_net(classes, model_name="", pretrain_weight_file ="",
         ctx=None, **kwargs):
    assert  model_name in ssd_base_models, "the model name is not supported, where the supported models are {}".format(ssd_base_models.keys())
    net = ssd_base_models[model_name](pretrained_base=False, ctx=ctx, **kwargs)
    net.load_parameters(pretrain_weight_file, ctx=ctx)
    # net.initilize()
    net.hybridize()
    return net


def _remap_keypoints(keypoints, padded_w, padded_h, expand, data_shape, ratio):
    """
    Remap bboxes in (x0, y0, x1, y1) format into the input image space
    Parameters
    ----------
    bboxes
    padded_w
    padded_h
    expand

    Returns
    -------

    """
    keypoints[:, 0::2] *= padded_w / (data_shape * ratio)
    keypoints[:, 1::2] *= padded_h / data_shape
    keypoints[:, 0::2] -= expand[0]
    keypoints[:, 1::2] -= expand[1]

    return keypoints


class GeneralDetector:

    def __init__(self, gpu_id,
                 aspect_ratio=1.,
                 data_shape=512,
                 model_name="ssd_512_mobilenet1.0_coco",
                 param_path="/home/ubuntu/.mxnet/models/ssd_512_mobilenet1.0_coco-da9756fa.params"
                 ):
        self.ctx = mx.gpu(gpu_id)

        self.net = get_net(classes = COCODetection.CLASSES,
                           ctx=mx.gpu(0),
                           model_name=model_name,
                           pretrain_weight_file = param_path)

        # self.net.hybridize()
        self.anchor_tensor = None
        self._anchor_image_shape = (1, 1)
        self._anchor_num = 1

        self.mean_mx = mx.nd.array(np.array([0.485, 0.456, 0.406])).as_in_context(self.ctx)
        self.std_mx = mx.nd.array(np.array([0.229, 0.224, 0.225])).as_in_context(self.ctx)
        self.ratio = aspect_ratio
        self.data_shape = data_shape

    def run_detection(self, image, tracking_box_indices, tracking_box_weights, tracking_box_classes):
        """

        Parameters
        ----------
        image: RGB images

        Returns
        -------

        """

        with timeit_context("preprocess"):
            data_tensor, padded_w, padded_h, expand = mxnet_frame_preprocessing(image, self.data_shape, self.ratio,
                                                                                self.mean_mx, self.std_mx, self.ctx)

            logging.info("input tensor shape {}".format(data_tensor.shape))
            mx.nd.waitall()

        # import pdb; pdb.set_trace()

        with timeit_context("network"):
            real_tracking_indices = tracking_box_indices + tracking_box_classes * self._anchor_num
            ids, scores, detection_bboxes, detection_anchor_indices, tracking_results, anchors = self.net(
                data_tensor.as_in_context(self.ctx), real_tracking_indices, tracking_box_weights)

            tracking_bboxes = tracking_results[:, [2, 3, 4, 5, 1]]

            detection_bboxes = _remap_bboxes(detection_bboxes[0, :, :],
                                             padded_w, padded_h, expand,
                                             self.data_shape, self.ratio)
            tracking_bboxes = _remap_bboxes(tracking_bboxes,
                                            padded_w, padded_h, expand,
                                            self.data_shape, self.ratio)
            mx.nd.waitall()
            # print("output shapes ids {} scores {} detection {} anchor {}".format(ids.shape, scores.shape, detection_bboxes.shape, anchors.shape))

        # import pdb; pdb.set_trace()

        # set anchors if needed
        if self._anchor_image_shape != (image.shape[:2]):
            self._anchor_image_shape = image.shape[:2]
            # initialize the anchor tensor for assignment
            self.anchor_tensor = anchors[0, :, :]
            half_w = self.anchor_tensor[:, 2] / 2
            half_h = self.anchor_tensor[:, 3] / 2
            center_x = self.anchor_tensor[:, 0].copy()
            center_y = self.anchor_tensor[:, 1].copy()

            # anchors are in the original format of (center_x, center_y, w, h)
            # translate them to (x0, y0, x1, y1)
            self.anchor_tensor[:, 0] = center_x - half_w
            self.anchor_tensor[:, 1] = center_y - half_h

            self.anchor_tensor[:, 2] = center_x + half_w
            self.anchor_tensor[:, 3] = center_y + half_h

            self.anchor_tensor = _remap_bboxes(self.anchor_tensor, padded_w, padded_h, expand,
                                               self.data_shape, self.ratio)
            self._anchor_num = self.anchor_tensor.shape[0]


        return ids[0], scores[0], detection_bboxes, tracking_bboxes, detection_anchor_indices[0].asnumpy()




if __name__ == '__main__':
    import cv2
    import numpy as np

    gpu=0

    image = cv2.imread('/home/ubuntu/data/shot_2/2/im00003797.jpg')
    image = cv2.resize(image, (512, 512))
    detector = GeneralDetector(0)

    indices = mx.nd.array([[1, 2], [2, 3]], dtype=int).as_in_context(mx.gpu(gpu))
    weights = mx.nd.array([[.5, .5], [.5, .5]], dtype=np.float32).as_in_context(mx.gpu(gpu))

    out = detector.run_detection(image[..., ::-1], indices, weights, 0)
    ids, scores, dets, tk_boxes, index = out


    scores = out[1]

    valid_det_num = (scores > 0.5).sum().astype(int).asnumpy()[0]


    if valid_det_num > 0:
        valid_scores = scores[:valid_det_num]
        valid_bboxes = dets[:valid_det_num, :]
        valid_classes = ids[:valid_det_num, :]
        # valid_kp = bbox_keypoints[:valid_det_num, :].asnumpy()
        detection_output = mx.nd.concat(valid_bboxes, valid_scores, valid_classes, dim=1).asnumpy()
        anchor_indices_output = index[:valid_det_num, :]

        print(detection_output)

