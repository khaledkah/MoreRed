from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

__all__ = ["DiffusionProcess"]


class DiffusionProcess(ABC):
    """
    Abstract base class to define diffusion processes.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        invariant: bool = True,
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            name: name of the diffusion process. Default is None.
            invariant: invariant: if True, force invariance to E(3) symmetries.
                        e.g. For atoms positions this would be to force a zero center.
            dtype: data type to use for computational accuracy.
        """
        self.name = name
        self.invariant = invariant
        self.dtype = dtype

        if isinstance(dtype, str):
            if dtype == "float64":
                self.dtype = torch.float64
            elif dtype == "float32":
                self.dtype = torch.float32
            else:
                raise ValueError(f"dtype must be float32 or float64, got {dtype}")

    @abstractmethod
    def get_T(self) -> int:
        """
        Returns the total number of diffusion steps T.
        """
        raise NotImplementedError

    def normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the time t to [0, 1].

        Args:
            t: time steps.
        """
        raise NotImplementedError

    def unnormalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Un-normalizes the time t to [0, T-1].

        Args:
            t: normalized time steps as float in [0, 1].
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
            idx_m: same as ``proporties.idx_m`` to assign each row to its system.
                    Set to None if one system or no invariance needed.
            t: time steps.
            return_dict: if True, return results under a dictionary of tensors.
            output_key: key to store the diffused x_t.
            kwargs: additional keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def reverse_step(
        self,
        x_t: torch.Tensor,
        model_out: Any,
        idx_m: Optional[torch.Tensor],
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Performs one reverse diffusion step to sample x_t-i ~ p(x_t-i|x_t), for i > 0.

        Args:
            x_t: input tensor x_t ~ p_t at diffusion step t.
            model_out: output of the denoiser model.
            idx_m: same as ``proporties.idx_m`` to assign each row of x to its system.
                Set to None if one system or no invariance needed.
            t: time steps.
            **kwargs: additional keyword arguments for subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_prior(
        self, x: torch.Tensor, idx_m: Optional[torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """
        Samples from the prior distribution p(x_T).
        The endpoint of the forward diffusion process
        and the starting point of the reverse process.

        Args:
            x: dummy input tensor, e.g. to infer shape.
            idx_m: same as ``proporties.idx_m`` to assign each row to its system.
            **kargs: additional keyword arguments.
        """
        raise NotImplementedError
