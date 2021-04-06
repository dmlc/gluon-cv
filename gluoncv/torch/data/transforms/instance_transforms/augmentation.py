# -*- coding: utf-8 -*-
# adapted from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/transforms/augmentation.py
import sys
import inspect
import random
import numpy as np
import pprint
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union
from PIL import Image

from .transform import (
    Transform,
    TransformList,
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    VFlipTransform,
    ExtentTransform,
    ResizeTransform,
    RotationTransform
)


__all__ = [
    "Augmentation",
    "TransformGen",
    "apply_transform_gens",
    "AugInput",
    "StandardAugInput",
    "apply_augmentations",
    "RandomApply",
    "RandomBrightness",
    "RandomContrast",
    "RandomCrop",
    "RandomExtent",
    "RandomFlip",
    "RandomSaturation",
    "RandomLighting",
    "RandomRotation",
    "Resize",
    "ResizeShortestEdge",
    "RandomCropWithInstance",
    "PadAugmentation",
]

"""
Overview of the augmentation system:

We have a design goal that aims at allowing:
    (1) Arbitrary structures of input data (e.g. list[list[boxes]], dict[str, boxes],
        multiple semantic segmentations for each image, etc) and arbitrary new data types
        (rotated boxes, 3D meshes, densepose, etc)
    (2) A list of augmentation to be applied sequentially

`Augmentation` defines policies to create deterministic transforms from input data.
An augmentation policy may need to access arbitrary input data, so it declares the input
data needed, to be provided by users when calling its `get_transform` method.

`Augmentation` is not able to apply transforms to data: data associated with
one sample may be much more than what `Augmentation` gets. For example, most
augmentation policies only need an image, but the actual input samples can be
much more complicated.

`AugInput` manages all inputs needed by `Augmentation` and implements the logic
to apply a sequence of augmentation. It has to define how the inputs are transformed,
because arguments needed by one `Augmentation` needs to be transformed to become arguments
of the next `Augmentation` in the sequence.

`AugInput` does not need to contain all input data, because most augmentation policies
only need very few fields (e.g., most only need "image"). We provide `StandardAugInput`
that only contains "images", "boxes", "sem_seg", that are enough to create transforms
for most cases. In this way, users keep the responsibility to apply transforms to other
potentially new data types and structures, e.g. keypoints, proposals boxes.

To extend the system, one can do:
1. To add a new augmentation policy that only needs to use standard inputs
   ("image", "boxes", "sem_seg"), writing a subclass of `Augmentation` is sufficient.
2. To use new data types or custom data structures, `StandardAugInput` can still be used as long
   as the new data types or custom data structures are not needed by any augmentation policy.
   The new data types or data structures can be transformed using the
   transforms returned by `AugInput.apply_augmentations`.
3. To add new augmentation policies that need new data types or data structures, in addition to
   implementing new `Augmentation`, a new `AugInput` is needed as well.
"""

def _check_img_dtype(img):
    assert isinstance(img, np.ndarray), "[Augmentation] Needs an numpy array, but got a {}!".format(
        type(img)
    )
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[Augmentation] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    assert img.ndim in [2, 3], img.ndim


class Augmentation(metaclass=ABCMeta):
    """
    Augmentation defines policies/strategies to generate :class:`Transform` from data.
    It is often used for pre-processing of input data. A policy typically contains
    randomness, but it can also choose to deterministically generate a :class:`Transform`.

    A "policy" that generates a :class:`Transform` may, in the most general case,
    need arbitrary information from input data in order to determine what transforms
    to apply. Therefore, each :class:`Augmentation` instance defines the arguments
    needed by its :meth:`get_transform` method with the :attr:`input_args` attribute.
    When called with the positional arguments defined by the :attr:`input_args`,
    the :meth:`get_transform` method executes the policy.

    Examples:
    ::
        # if a policy needs to know both image and semantic segmentation
        assert aug.input_args == ("image", "sem_seg")
        tfm: Transform = aug.get_transform(image, sem_seg)
        new_image = tfm.apply_image(image)

    To implement a custom :class:`Augmentation`, define its :attr:`input_args` and
    implement :meth:`get_transform`.

    Note that :class:`Augmentation` defines the policies to create a :class:`Transform`,
    but not how to apply the actual transform to those data.
    """

    input_args: Tuple[str] = ("image",)
    """
    Attribute of class instances that defines the argument(s) needed by
    :meth:`get_transform`. Default to only "image", because most policies only
    require knowing the image in order to determine the transform.

    Users can freely define arbitrary new args and their types in custom
    :class:`Augmentation`. In detectron2 we use the following convention:

    * image: (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
      floating point in range [0, 1] or [0, 255].
    * boxes: (N,4) ndarray of float32. It represents the instance bounding boxes
      of N instances. Each is in XYXY format in unit of absolute coordinates.
    * sem_seg: (H,W) ndarray of type uint8. Each element is an integer label of pixel.

    We do not specify convention for other types and do not include builtin
    :class:`Augmentation` that uses other types in detectron2.
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    # NOTE: in the future, can allow it to return list[Augmentation],
    # to delegate augmentation to others
    @abstractmethod
    def get_transform(self, *args) -> Transform:
        """
        Execute the policy to use input data to create transform(s).

        Args:
            arguments must follow what's defined in :attr:`input_args`.

        Returns:
            Return a :class:`Transform` instance, which is the transform to apply to inputs.
        """
        pass

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyAugmentation(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(name)
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__


TransformGen = Augmentation
"""
Alias for Augmentation, since it is something that generates :class:`Transform`s
"""


class AugInput:
    """
    A base class for anything on which a list of :class:`Augmentation` can be applied.
    This class provides input arguments for :class:`Augmentation` to use, and defines how
    to apply transforms to these data.

    An instance of this class must satisfy the following:

    * :class:`Augmentation` declares some data it needs as arguments. A :class:`AugInput`
      must provide access to these data in the form of attribute access (``getattr``).
      For example, if a :class:`Augmentation` to be applied needs "image" and "sem_seg"
      arguments, this class must have the attribute "image" and "sem_seg" whose content
      is as required by the :class:`Augmentation`s.
    * This class must have a :meth:`transform(tfm: Transform) -> None` method which
      in-place transforms all attributes stored in the class.
    """

    def transform(self, tfm: Transform) -> None:
        raise NotImplementedError

    def apply_augmentations(
        self, augmentations: List[Union[Augmentation, Transform]]
    ) -> TransformList:
        """
        Apply a list of Transform/Augmentation in-place and returned the applied transform.
        Attributes of this class will be modified.

        Returns:
            TransformList:
                returns transformed inputs and the list of transforms applied.
                The TransformList can then be applied to other data associated with the inputs.
        """
        tfms = []
        for aug in augmentations:
            if isinstance(aug, Augmentation):
                args = []
                for f in aug.input_args:
                    try:
                        args.append(getattr(self, f))
                    except AttributeError:
                        raise AttributeError(
                            f"Augmentation {aug} needs '{f}', which is not an attribute of {self}!"
                        )

                tfm = aug.get_transform(*args)
                assert isinstance(tfm, Transform), (
                    f"{type(aug)}.get_transform must return an instance of Transform! "
                    "Got {type(tfm)} instead."
                )
            else:
                tfm = aug
            self.transform(tfm)
            tfms.append(tfm)
        return TransformList(tfms)


class StandardAugInput(AugInput):
    """
    A standard implementation of :class:`AugInput` for the majority of use cases.
    This class provides the following standard attributes that are common to use by
    Augmentation (augmentation policies). These are chosen because most
    :class:`Augmentation` won't need anything more to define a augmentation policy.
    After applying augmentations to these special attributes, the returned transforms
    can then be used to transform other data structures that users have.

    Attributes:
        image (ndarray): image in HW or HWC format. The meaning of C is up to users
        boxes (ndarray or None): Nx4 boxes in XYXY_ABS mode
        sem_seg (ndarray or None): HxW semantic segmentation mask

    Examples:
    ::
        input = StandardAugInput(image, boxes=boxes)
        tfms = input.apply_augmentations(list_of_augmentations)
        transformed_image = input.image
        transformed_boxes = input.boxes
        transformed_other_data = tfms.apply_other(other_data)

    An extended project that works with new data types may require augmentation
    policies that need more inputs. An algorithm may need to transform inputs
    in a way different from the standard approach defined in this class. In those
    situations, users can implement new subclasses of :class:`AugInput` with differnt
    attributes and the :meth:`transform` method.
    """

    def __init__(
        self,
        image: np.ndarray,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray] = None,
    ):
        """
        Args:
            image: (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
                floating point in range [0, 1] or [0, 255].
            boxes: (N,4) ndarray of float32. It represents the instance bounding boxes
                of N instances. Each is in XYXY format in unit of absolute coordinates.
            sem_seg: (H,W) ndarray of type uint8. Each element is an integer label of pixel.
        """
        _check_img_dtype(image)
        self.image = image
        self.boxes = boxes
        self.sem_seg = sem_seg

    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)


def apply_augmentations(augmentations: List[Union[Transform, Augmentation]], inputs):
    """
    Use :meth:`AugInput.apply_augmentations` instead.
    """
    if isinstance(inputs, np.ndarray):
        # handle the common case of image-only Augmentation, also for backward compatibility
        image_only = True
        inputs = StandardAugInput(inputs)
    else:
        image_only = False
    tfms = inputs.apply_augmentations(augmentations)
    return inputs.image if image_only else inputs, tfms


apply_transform_gens = apply_augmentations
"""
Alias for backward-compatibility.
"""

class RandomApply(Augmentation):
    """
    Randomly apply the wrapper transformation with a given probability.
    """

    def __init__(self, transform, prob=0.5):
        """
        Args:
            transform (Transform, Augmentation): the transform to be wrapped
                by the `RandomApply`. The `transform` can either be a
                `Transform` or `Augmentation` instance.
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
        """
        super().__init__()
        assert isinstance(transform, (Transform, Augmentation)), (
            f"The given transform must either be a Transform or Augmentation instance. "
            f"Not {type(transform)}"
        )
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        self.prob = prob
        self.transform = transform
        if isinstance(transform, Augmentation):
            self.input_args = transform.input_args

    def get_transform(self, img):
        do = self._rand_range() < self.prob
        if do:
            if isinstance(self.transform, Augmentation):
                return self.transform.get_transform(img)
            else:
                return self.transform
        else:
            return NoOpTransform()


class RandomFlip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


class Resize(Augmentation):
    """ Resize image to a fixed target size"""

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, img):
        return ResizeTransform(
            img.shape[0], img.shape[1], self.shape[0], self.shape[1], self.interp
        )


class ResizeShortestEdge(Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)


class RandomRotation(Augmentation):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around the given center.
    """

    def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        center = None
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        if angle % 360 == 0:
            return NoOpTransform()

        return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)


class RandomCrop(Augmentation):
    """
    Randomly crop a subimage out of an image.
    """

    def __init__(self, crop_type: str, crop_size):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
                See `config/defaults.py` for explanation.
            crop_size (tuple[float]): the relative ratio or absolute pixels of
                height and width
        """
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


class RandomExtent(Augmentation):
    """
    Outputs an image by cropping a random "subrect" of the source image.

    The subrect can be parameterized to include pixels outside the source image,
    in which case they will be set to zeros (i.e. black). The size of the output
    image will vary with the size of the random subrect.
    """

    def __init__(self, scale_range, shift_range):
        """
        Args:
            output_size (h, w): Dimensions of output image
            scale_range (l, h): Range of input-to-output size scaling factor
            shift_range (x, y): Range of shifts of the cropped subrect. The rect
                is shifted by [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)],
                where (w, h) is the (width, height) of the input image. Set each
                component to zero to crop at the image's center.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        img_h, img_w = img.shape[:2]

        # Initialize src_rect to fit the input image.
        src_rect = np.array([-0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h])

        # Apply a random scaling to the src_rect.
        src_rect *= np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Apply a random shift to the coordinates origin.
        src_rect[0::2] += self.shift_range[0] * img_w * (np.random.rand() - 0.5)
        src_rect[1::2] += self.shift_range[1] * img_h * (np.random.rand() - 0.5)

        # Map src_rect coordinates into image coordinates (center at corner).
        src_rect[0::2] += 0.5 * img_w
        src_rect[1::2] += 0.5 * img_h

        return ExtentTransform(
            src_rect=(src_rect[0], src_rect[1], src_rect[2], src_rect[3]),
            output_size=(int(src_rect[3] - src_rect[1]), int(src_rect[2] - src_rect[0])),
        )


class RandomContrast(Augmentation):
    """
    Randomly transforms image contrast.

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=img.mean(), src_weight=1 - w, dst_weight=w)


class RandomBrightness(Augmentation):
    """
    Randomly transforms image brightness.

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)


class RandomSaturation(Augmentation):
    """
    Randomly transforms saturation of an RGB image.
    Input images are assumed to have 'RGB' channel order.

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation (1 preserves input).
            intensity_max (float): Maximum augmentation (1 preserves input).
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        assert img.shape[-1] == 3, "RandomSaturation only works on RGB images"
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = img.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)


class RandomLighting(Augmentation):
    """
    The "lighting" augmentation described in AlexNet, using fixed PCA over ImageNet.
    Input images are assumed to have 'RGB' channel order.

    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    """

    def __init__(self, scale):
        """
        Args:
            scale (float): Standard deviation of principal component weighting.
        """
        super().__init__()
        self._init(locals())
        self.eigen_vecs = np.array(
            [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

    def get_transform(self, img):
        assert img.shape[-1] == 3, "RandomLighting only works on RGB images"
        weights = np.random.normal(scale=self.scale, size=3)
        return BlendTransform(
            src_image=self.eigen_vecs.dot(weights * self.eigen_vals), src_weight=1.0, dst_weight=1.0
        )

def _gen_crop_transform_with_instance(crop_size, image_size, instances, crop_box=True):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    bbox = random.choice(instances)
    crop_size = np.asarray(crop_size, dtype=np.int32)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)

    # if some instance is cropped extend the box
    if not crop_box:
        num_modifications = 0
        modified = True

        # convert crop_size to float
        crop_size = crop_size.astype(np.float32)
        while modified:
            modified, x0, y0, crop_size = adjust_crop(x0, y0, crop_size, instances)
            num_modifications += 1
            if num_modifications > 100:
                raise ValueError(
                    "Cannot finished cropping adjustment within 100 tries (#instances {}).".format(
                        len(instances)
                    )
                )
                return CropTransform(0, 0, image_size[1], image_size[0])

    return CropTransform(*map(int, (x0, y0, crop_size[1], crop_size[0])))


def adjust_crop(x0, y0, crop_size, instances, eps=1e-3):
    modified = False

    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]

    for bbox in instances:

        if bbox[0] < x0 - eps and bbox[2] > x0 + eps:
            crop_size[1] += x0 - bbox[0]
            x0 = bbox[0]
            modified = True

        if bbox[0] < x1 - eps and bbox[2] > x1 + eps:
            crop_size[1] += bbox[2] - x1
            x1 = bbox[2]
            modified = True

        if bbox[1] < y0 - eps and bbox[3] > y0 + eps:
            crop_size[0] += y0 - bbox[1]
            y0 = bbox[1]
            modified = True

        if bbox[1] < y1 - eps and bbox[3] > y1 + eps:
            crop_size[0] += bbox[3] - y1
            y1 = bbox[3]
            modified = True

    return modified, x0, y0, crop_size


class RandomCropWithInstance(RandomCrop):
    """ Instance-aware cropping.
    """

    def __init__(self, crop_type, crop_size, crop_instance=True):
        """
        Args:
            crop_instance (bool): if False, extend cropping boxes to avoid cropping instances
        """
        super().__init__(crop_type, crop_size)
        self.crop_instance = crop_instance
        self.input_args = ("image", "boxes")

    def get_transform(self, img, boxes):
        image_size = img.shape[:2]
        crop_size = self.get_crop_size(image_size)
        return _gen_crop_transform_with_instance(
            crop_size, image_size, boxes, crop_box=self.crop_instance
        )


class PadAugmentation(Augmentation):
    def __init__(self, crop_size):
        """
        Args:
            crop_instance (bool): if False, extend cropping boxes to avoid cropping instances
        """
        super().__init__()
        self.crop_size = crop_size

    def get_crop_size(self, image_size):
        h, w = image_size
        return (min(self.crop_size[0], h), min(self.crop_size[1], w))

    def get_transform(self, img):
        image_size = img.shape[:2]
        image_size = self.get_crop_size(image_size)
        return _PadTransform(image_size[0], image_size[1], self.crop_size[1], self.crop_size[0])


class _PadTransform(Transform):
    def __init__(self, h: int, w: int, crop_h: int, crop_w: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:

        h, w = img.shape[:2]
        assert (
                self.h == h and self.w == w
        ), "Input size mismatch h w {}:{} -> {}:{}".format(self.h, self.w, h, w)

        padding = ((0, self.crop_h - h), (0, self.crop_w - w), (0, 0))
        img = np.pad(img, pad_width=padding)

        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords
