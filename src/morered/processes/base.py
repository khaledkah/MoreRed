from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

__all__ = ["ForwardDiffusion"]


class ForwardDiffusion(ABC):
    """
    Abstract base class to define the forward diffusion processes.
    """

    def __init__(
        self,
        invariant: bool = False,
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            invariant: invariant: if True, apply invariance constraints for symmetries.
                        e.g. For atoms positions this would be to force a zero CoG.
            dtype: data type to use for computational accuracy.
        """
        self.invariant = invariant
        self.dtype = dtype

        if isinstance(dtype, str):
            if dtype == "float64":
                self.dtype = torch.float64
            elif dtype == "float32":
                self.dtype = torch.float32
            else:
                raise ValueError(f"data type must be float32 or float64, got {dtype}")

    @abstractmethod
    def sample_prior(
        self, x: torch.Tensor, idx_m: Optional[torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """
        Samples from the prior distribution p(x_T).

        Args:
            x: input tensor, e.g. to infer shape.
            idx_m: same as ``proporties.idx_m`` to map each row to its system.
            **kargs: additional keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def diffuse(
        self,
        x_0: torch.Tensor,
        idx_m: Optional[torch.Tensor],
        t: torch.Tensor,
        return_dict: bool = False,
        output_key: str = "x_t",
        **kwargs,
    ) -> Any:
        """
        Diffuses origin x_0 by t steps to sample x_t from p(x_t|x_0),
        given x_0 ~ p_data.

        Args:
            x_0: input tensor x_0 ~ p_data to be diffused.
            idx_m: same as ``proporties.idx_m`` to map each row to its system.
                Set to None if one system or no invariance needed.
            t: time steps.
            return_dict: if True, return results under a dictionary of tensors.
            output_key: key to store the diffused x_t.
            kwargs: additional keyword arguments.
        """
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Diffuse is default object call.
        """
        return self.diffuse(*args, **kwargs)
