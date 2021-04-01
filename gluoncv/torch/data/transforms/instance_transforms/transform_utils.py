# adapted from https://github.com/facebookresearch/fvcore/blob/master/fvcore/transforms/transform_util.py

import numpy as np
import torch

def to_float_tensor(numpy_array: np.ndarray) -> torch.Tensor:
    """
    Convert the numpy array to torch float tensor with dimension of NxCxHxW.
    Pytorch is not fully supporting uint8, so convert tensor to float if the
    numpy_array is uint8.
    Args:
        numpy_array (ndarray): of shape NxHxWxC, or HxWxC or HxW to
            represent an image. The array can be of type uint8 in range
            [0, 255], or floating point in range [0, 1] or [0, 255].
    Returns:
        float_tensor (tensor): converted float tensor.
    """
    assert isinstance(numpy_array, np.ndarray)
    assert len(numpy_array.shape) in (2, 3, 4)

    # Some of the input numpy array has negative strides. Pytorch currently
    # does not support negative strides, perform ascontiguousarray to
    # resolve the issue.
    float_tensor = torch.from_numpy(np.ascontiguousarray(numpy_array))
    if numpy_array.dtype in (np.uint8, np.int32, np.int64):
        float_tensor = float_tensor.float()

    if len(numpy_array.shape) == 2:
        # HxW -> 1x1xHxW.
        float_tensor = float_tensor[None, None, :, :]
    elif len(numpy_array.shape) == 3:
        # HxWxC -> 1xCxHxW.
        float_tensor = float_tensor.permute(2, 0, 1)
        float_tensor = float_tensor[None, :, :, :]
    elif len(numpy_array.shape) == 4:
        # NxHxWxC -> NxCxHxW
        float_tensor = float_tensor.permute(0, 3, 1, 2)
    else:
        raise NotImplementedError(
            "Unknow numpy_array dimension of {}".format(float_tensor.shape)
        )
    return float_tensor


def to_numpy(
    float_tensor: torch.Tensor, target_shape: list, target_dtype: np.dtype
) -> np.ndarray:
    """
    Convert float tensor with dimension of NxCxHxW back to numpy array.
    Args:
        float_tensor (tensor): a float pytorch tensor with shape of NxCxHxW.
        target_shape (list): the target shape of the numpy array to represent
            the image as output. options include NxHxWxC, or HxWxC or HxW.
        target_dtype (dtype): the target dtype of the numpy array to represent
            the image as output. The array can be of type uint8 in range
            [0, 255], or floating point in range [0, 1] or [0, 255].
    Returns:
        (ndarray): converted numpy array.
    """
    assert len(target_shape) in (2, 3, 4)

    if len(target_shape) == 2:
        # 1x1xHxW -> HxW.
        assert float_tensor.shape[0] == 1
        assert float_tensor.shape[1] == 1
        float_tensor = float_tensor[0, 0, :, :]
    elif len(target_shape) == 3:
        assert float_tensor.shape[0] == 1
        # 1xCxHxW -> HxWxC.
        float_tensor = float_tensor[0].permute(1, 2, 0)
    elif len(target_shape) == 4:
        # NxCxHxW -> NxHxWxC
        float_tensor = float_tensor.permute(0, 2, 3, 1)
    else:
        raise NotImplementedError(
            "Unknow target shape dimension of {}".format(target_shape)
        )
    if target_dtype == np.uint8:
        # Need to specifically call round here, notice in pytroch the round
        # is half to even.
        # https://github.com/pytorch/pytorch/issues/16498
        float_tensor = float_tensor.round().byte()
    return float_tensor.numpy()
