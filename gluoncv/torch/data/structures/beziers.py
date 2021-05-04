from typing import Union
import torch

class Beziers:
    """
    This structure stores a list of bezier curves as a Nx16 torch.Tensor.
    It will support some common methods about bezier shapes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, 16)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 16, tensor.size()

        self.tensor = tensor

    def to(self, device: str) -> "Beziers":
        return Beziers(self.tensor.to(device))

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Beziers":
        """
        Returns:
            Beziers: Create a new :class:`Beziers` by indexing.
        """
        if isinstance(item, int):
            return Beziers(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Beziers(b)
