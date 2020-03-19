""" tracking Data augmentation

Code adapted from https://github.com/STVIR/pysot
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from gluoncv.utils.filesystem import try_import_cv2
from gluoncv.model_zoo.siamrpn.siamrpn_tracker import corner2center, center2corner
from gluoncv.model_zoo.siamrpn.siamrpn_tracker import Center, Corner
class SiamRPNaugmentation:
    """dataset Augmentation for SiamRPN tracking.

    Parameters
    ----------
    shift : int
        length of template augmentation shift
    scale : float
        template augmentation scale ratio
    blur : float
        template augmentation blur ratio
    flip : float
        template augmentation flip ratio
    color : float
        template augmentation color ratio
    """
    def __init__(self, shift, scale, blur, flip, color):
        self.shift = shift
        self.scale = scale
        self.blur = blur
        self.flip = flip
        self.color = color
        self.rgbVar = np.array([[-0.55919361, 0.98062831, - 0.41940627],
                                [1.72091413, 0.19879334, - 1.82968581],
                                [4.64467907, 4.73710203, 4.88324118]], dtype=np.float32)
        self.cv2 = try_import_cv2()

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    def _crop_roi(self, image, bbox, out_sz, padding=(0, 0, 0)):
        """crop image roi size.

        Parameters
        ----------
        image : np.array
            image
        bbox : list or np.array
            bbox coordinate，like (xmin,ymin,xmax,ymax)
        out_sz : int
            size after crop

        Return:
            image after crop
        """
        bbox = [float(x) for x in bbox]
        a = (out_sz-1) / (bbox[2]-bbox[0])
        b = (out_sz-1) / (bbox[3]-bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = self.cv2.warpAffine(image, mapping, (out_sz, out_sz),
                                   borderMode=self.cv2.BORDER_CONSTANT,
                                   borderValue=padding)
        return crop

    def _blur_aug(self, image):
        """blur filter to smooth image

        Parameters
        ----------
        image : np.array
            image

        Return:
            image after blur
        """
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel
        kernel = rand_kernel()
        image = self.cv2.filter2D(image, -1, kernel)
        return image

    def _color_aug(self, image):
        """Random increase of image channel

        Parameters
        ----------
        image : np.array
            image

        Return:
            image after Random increase of channel
        """
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset
        return image

    def _gray_aug(self, image):
        """image Grayscale

        Parameters
        ----------
        image : np.array
            image

        Return:
            image after Grayscale
        """
        grayed = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        image = self.cv2.cvtColor(grayed, self.cv2.COLOR_GRAY2BGR)
        return image

    def _shift_scale_aug(self, image, bbox, crop_bbox, size):
        """shift scale augmentation

        Parameters
        ----------
        image : np.array
            image
        bbox : list or np.array
            bbox
        crop_bbox :
            crop size image from center

        Return
            image ,bbox after shift and scale
        """
        im_h, im_w = image.shape[:2]

        # adjust crop bounding box
        crop_bbox_center = corner2center(crop_bbox)

        if self.scale:
            scale_x = (1.0 + SiamRPNaugmentation.random() * self.scale)
            scale_y = (1.0 + SiamRPNaugmentation.random() * self.scale)
            h, w = crop_bbox_center.h, crop_bbox_center.w
            scale_x = min(scale_x, float(im_w) / w)
            scale_y = min(scale_y, float(im_h) / h)
            crop_bbox_center = Center(crop_bbox_center.x,
                                      crop_bbox_center.y,
                                      crop_bbox_center.w * scale_x,
                                      crop_bbox_center.h * scale_y)

        crop_bbox = center2corner(crop_bbox_center)
        if self.shift:
            sx = SiamRPNaugmentation.random() * self.shift
            sy = SiamRPNaugmentation.random() * self.shift

            x1, y1, x2, y2 = crop_bbox

            sx = max(-x1, min(im_w - 1 - x2, sx))
            sy = max(-y1, min(im_h - 1 - y2, sy))

            crop_bbox = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)

        # adjust target bounding box
        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
                      bbox.x2 - x1, bbox.y2 - y1)

        if self.scale:
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
                          bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_roi(image, crop_bbox, size)
        return image, bbox

    def _flip_aug(self, image, bbox):
        """flip augmentation

        Parameters
        ----------
        image : np.array
            image
        bbox : list or np.array
            bbox

        Return
            image and bbox after filp
        """
        image = self.cv2.flip(image, 1)
        width = image.shape[1]
        bbox = Corner(width - 1 - bbox.x2, bbox.y1,
                      width - 1 - bbox.x1, bbox.y2)
        return image, bbox

    def __call__(self, image, bbox, size, gray=False):
        shape = image.shape
        crop_bbox = center2corner(Center(shape[0]//2, shape[1]//2,
                                         size-1, size-1))
        # gray augmentation
        if gray:
            image = self._gray_aug(image)

        # shift scale augmentation
        image, bbox = self._shift_scale_aug(image, bbox, crop_bbox, size)

        # color augmentation
        if self.color > np.random.random():
            image = self._color_aug(image)

        # blur augmentation
        if self.blur > np.random.random():
            image = self._blur_aug(image)

        # flip augmentation
        if self.flip and self.flip > np.random.random():
            image, bbox = self._flip_aug(image, bbox)
        return image, bbox
