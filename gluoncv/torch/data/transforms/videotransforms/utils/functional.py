def normalize(tensor, mean, std):
    """
    Args:
        tensor (Tensor): Tensor to normalize

    Returns:
        Tensor: Normalized tensor
    """
    tensor.sub_(mean).div_(std)
    return tensor
