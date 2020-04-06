# pylint: disable=arguments-differ,line-too-long,missing-docstring,missing-module-docstring
import math
import random

# code adapted from:
# https://github.com/kakaobrain/fast-autoaugment
# https://github.com/rpmcruz/autoaugment
class EfficientNetRandomCrop:
    # pylint: disable=misplaced-comparison-constant
    def __init__(self, imgsize, min_covered=0.1, aspect_ratio_range=(3./4, 4./3),
                 area_range=(0.1, 1.0), max_attempts=10):
        assert 0.0 < min_covered
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]
        assert 0 < area_range[0] <= area_range[1]
        assert 1 <= max_attempts

        self.min_covered = min_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self._fallback = EfficientNetCenterCrop(imgsize)

    def __call__(self, img):
        # https://github.com/tensorflow/tensorflow/blob/9274bcebb31322370139467039034f8ff852b004/tensorflow/core/kernels/sample_distorted_bounding_box_op.cc#L111
        original_width, original_height = img.size
        min_area = self.area_range[0] * (original_width * original_height)
        max_area = self.area_range[1] * (original_width * original_height)

        for _ in range(self.max_attempts):
            aspect_ratio = random.uniform(*self.aspect_ratio_range)
            height = int(round(math.sqrt(min_area / aspect_ratio)))
            max_height = int(round(math.sqrt(max_area / aspect_ratio)))

            if max_height * aspect_ratio > original_width:
                max_height = (original_width + 0.5 - 1e-7) / aspect_ratio
                max_height = int(max_height)
                if max_height * aspect_ratio > original_width:
                    max_height -= 1

            if max_height > original_height:
                max_height = original_height

            if height >= max_height:
                height = max_height

            height = int(round(random.uniform(height, max_height)))
            width = int(round(height * aspect_ratio))
            area = width * height

            if area < min_area or area > max_area:
                continue
            if width > original_width or height > original_height:
                continue
            if area < self.min_covered * (original_width * original_height):
                continue
            if width == original_width and height == original_height:
                return self._fallback(img)      # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L102

            x = random.randint(0, original_width - width)
            y = random.randint(0, original_height - height)
            return img.crop((x, y, x + width, y + height))

        return self._fallback(img)


class EfficientNetCenterCrop:
    def __init__(self, imgsize):
        self.imgsize = imgsize

    def __call__(self, img):
        """Crop the given PIL Image and resize it to desired size.
        Args:
            img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
        Returns:
            PIL Image: Cropped image.
        """
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
