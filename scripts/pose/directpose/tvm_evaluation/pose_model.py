import tvm
from tvm.contrib import graph_runtime
from typing import List
import cv2
import numpy as np
try:
    import torch
except ImportError:
    torch = None

class PoseEstimationInferenceModel():
    def __init__(
            self,
            model_prefix,
            gpu_id,
            image_width=1280,
            image_height=720,
            threshold=0.05,
            use_torch=False
    ):
        self.torch = use_torch
        if self.torch:
            assert torch is not None, "requires torch when `use_torch` is enabled"
        self.__new_nd_array = tvm.nd.array
        self.device_ctx = tvm.gpu(gpu_id)
        self.image_width = image_width
        self.image_height = image_height
        self.__model = self.load_model(model_prefix, self.device_ctx)
        self.threshold = threshold
        self.mean = np.array([0.406 * 255, 0.456 * 255, 0.485 * 255], dtype=np.float32)
        self.std = np.array([1, 1, 1], dtype=np.float32)

    def load_model(self, model_prefix, device_ctx):
        if not self.torch:
            with open('{}.json'.format(model_prefix), 'r') as f:
                model_json = f.read()
            model_lib = tvm.runtime.load_module('{}.so'.format(model_prefix))
            with open('{}.params'.format(model_prefix), "rb") as f:
                model_params = bytearray(f.read())
            module = graph_runtime.create(
                model_json,
                model_lib,
                device_ctx
            )
            module.load_params(model_params)
            return module
        import torch
        from gluoncv.torch import model_zoo
        from gluoncv.torch.engine.config import get_cfg_defaults
        torch.set_grad_enabled(False)
        device = torch.device('cuda')
        cfg = get_cfg_defaults(name='directpose')
        # cfg.merge_from_file('./configurations/ms_dla_34_4x_syncbn.yaml')
        # net = model_zoo.dla34_fpn_directpose(cfg).to(device).eval()
        # model = torch.load('model_final.pth')['model']
        cfg.merge_from_file('./configurations/ms_aa_resnet50_4x_syncbn.yaml')
        net = model_zoo.directpose_resnet_lpf_fpn(cfg).to(device).eval()
        model = torch.load('model_final_resnet.pth')['model']
        # _ = net(torch.zeros((1,3, self.image_height, self.image_width)).cuda())
        net.load_state_dict(model, strict=False)
        return net

    def process(
            self,
            payloads: List = [],
            **kwargs
        ) -> List[object]:
        """
        :param payloads: a list of {
                                    "image": RGB channel raw image or pre-processed tensor
                                    "transform_info": None means "image" is RGB channel raw image, otherwise "image" is pre-processed tensor and this if the transformation info
                                    }
        :return:
        for each payload in payloads:
            preprocessed_payload = __preprocess(payload)
            results = run PersonDetection against preprocessed_payload
            postprocessed_results = __postprocess(results)
        the final results will be a list of {
                                                "bboxes": N x 4 numpy array in format (x0, y0, x1, y1),
                                                "class_ids": N x 1 numpy array contaning class ids for each bounding box, id = 0 means person,
                                                "scores": N x 1 numpy array containing confidence score for each bounding box
                                            }
        """
        results = []
        for image in payloads:
            if image["transform_info"] is None:
                img, transform_info = self.__preprocess(image["image"])
            else:   # image already preprocessed
                img = image["image"]
                transform_info = image["transform_info"]

            if not self.torch:
                self.__model.set_input("input0", self.__new_nd_array(img, ctx=self.device_ctx))
                self.__model.run()
                class_IDs, nms_ret, bounding_boxs, scores, keypoints = self.__model.get_output(0), self.__model.get_output(1), \
                                                self.__model.get_output(2), self.__model.get_output(3), self.__model.get_output(4)
                # dd = class_IDs.asnumpy()
                # print(dd.shape, dd)
                # raise
                np_bbox, np_ids, np_scores, np_kpts = self. __postprocess(bounding_boxs, class_IDs, scores, keypoints, transform_info, nms_ret)
            else:
                # img = np.load('input.npy')
                class_IDs, idxs, bounding_boxs, scores, keypoints = self.__model(torch.as_tensor(img).cuda())
                # dd = bounding_boxs.cpu().numpy()
                # print(dd.shape, dd)
                # raise
                np_bbox, np_ids, np_scores, np_kpts = self. __postprocess(bounding_boxs, class_IDs, scores, keypoints, transform_info, idxs)


            results.append({
                "bboxes": np_bbox,
                "class_ids": np_ids,
                "scores": np_scores,
                "keypoints": np_kpts
            })
            # self.visualize(np_bbox, np_scores, image["image"], transform_info)
            #print(np_bbox.shape, np_ids.shape, np_scores.shape, np_kpts.shape)
        return results

    def __preprocess(
            self,
            payload: object = None
    ):
        """
        :param payload: RGB channel
        :return: pre-processed image, and transform info (p_b_w, p_b_h, pad_size_h, pad_size_w)
        """
        self.original_h, self.original_w, _ = payload.shape
        if float(self.original_h) / self.original_w >= float(self.image_height) / float(self.image_width):
            resize_ratio = (float(self.original_h) * self.image_width) / (float(self.original_w) * self.image_height)
            pad_size = (self.original_h, int(self.original_w * resize_ratio))
        else:
            resize_ratio = (float(self.original_w) * self.image_height) / (float(self.original_h) * self.image_width)
            pad_size = (int(self.original_h * resize_ratio), self.original_w)

        p_b_h = (pad_size[0] - self.original_h) // 2
        p_b_w = (pad_size[1] - self.original_w) // 2
        p_a_h = pad_size[0] - self.original_h - p_b_h
        p_a_w = pad_size[1] - self.original_w - p_b_w

        pad_width = ((p_b_h, p_a_h), (p_b_w, p_a_w), (0, 0))
        transform_info = (p_b_w, p_b_h, pad_size[0], pad_size[1])

        image = payload.astype(np.float32)


        # noramlize the image
        image = (image - self.mean) / self.std

        # image padding
        image = np.pad(image, pad_width)

        # image resize (for opencv, width come first)
        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)

        # noramlize the image
        # image = (image - self.mean) / self.std

        # convert to CxHXW
        image = np.transpose(image, (2, 0, 1))

        return image[None, ...], transform_info

    def __postprocess(
            self,
            bounding_boxs,
            class_IDs,
            scores,
            keypoints,
            transform_info,
            nms_ret=None
    ):
        """
        post processing the results[] gotten from process
        """

        p_b_w, p_b_h, padded_h, padded_w = transform_info
        assert nms_ret is not None
        if self.torch:
            idxs = nms_ret
            np_bbx = bounding_boxs[idxs].cpu().numpy()
            np_scores =  scores[idxs].cpu().numpy()
            np_ids = class_IDs[idxs].cpu().numpy()
            np_keypoints = keypoints[idxs].cpu().numpy()
        else:
            np_bbx = bounding_boxs.asnumpy()
            np_ids = class_IDs.asnumpy()
            np_scores = scores.asnumpy()
            np_keypoints = keypoints.asnumpy()
            idxs = nms_ret.asnumpy()[:, -1].astype(int)
            idxs = idxs[np.where(idxs >= 0)[0]]
            np_bbx = np_bbx[idxs, :]
            np_ids = np_ids[idxs]
            np_scores = np_scores[idxs]
            np_keypoints = np_keypoints[idxs, :, :]
            # print(np_scores.shape, np_bbx.shape, np_keypoints.shape, np_ids.shape)

        idx = np.where(np_scores > self.threshold)[0]
        np_bbx = np_bbx[idx, :]
        np_scores = np_scores[idx, np.newaxis]
        np_ids = np_ids[idx]
        np_keypoints = np_keypoints[idx, :, :]

        # idx = np.where(np_scores > self.threshold)[0]
        # np_bbox = np_bbx[idx, :]
        # np_scores = np_scores[idx]
        # np_ids = np_ids[idx]
        # np_keypoints = np_keypoints[idx, :, :]

        np_bbx[:, 0] *= padded_w / self.image_width
        np_bbx[:, 1] *= padded_h / self.image_height
        np_bbx[:, 2] *= padded_w / self.image_width
        np_bbx[:, 3] *= padded_h / self.image_height

        np_bbx[:, 0] -= p_b_w
        np_bbx[:, 1] -= p_b_h
        np_bbx[:, 2] -= p_b_w
        np_bbx[:, 3] -= p_b_h

        np_bbx[:, 0] = np.clip(np_bbx[:, 0], a_min=0, a_max=self.original_w)
        np_bbx[:, 2] = np.clip(np_bbx[:, 2], a_min=0, a_max=self.original_w)
        np_bbx[:, 1] = np.clip(np_bbx[:, 1], a_min=0, a_max=self.original_h)
        np_bbx[:, 3] = np.clip(np_bbx[:, 3], a_min=0, a_max=self.original_h)

        np_keypoints = np_keypoints.reshape(-1, 17, 3)

        np_keypoints[:, :, 0] *= padded_w / self.image_width
        np_keypoints[:, :, 1] *= padded_h / self.image_height

        np_keypoints[:, :, 0] -= p_b_w
        np_keypoints[:, :, 1] -= p_b_h

        np_keypoints[:, :, 0] = np_keypoints[:, :, 0].clip(0, self.original_w)
        np_keypoints[:, :, 1] = np_keypoints[:, :, 1].clip(0, self.original_h)
        np_keypoints[:, :, 2] = np_keypoints[:, :, 2].clip(0, 1)

        return np_bbx, np_ids, np_scores, np_keypoints
