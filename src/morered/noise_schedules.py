import logging
from typing import Dict, Union

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)
__all__ = ["CosineSchedule", "PolynomialSchedule", "NoiseSchedule"]


def clip_noise_schedule(
    alphas_bar: np.ndarray, clip_min: float = 0.0, clip_max: float = 1.0
) -> np.ndarray:
    """
    For a noise schedule given by alpha_bar, this clips alpha_t / alpha_t-1.
    This may help improve stability during sampling.

    Args:
        alphas_bar: noise schedule.
        clip_min: minimum value to clip to.
        clip_max: maximum value to clip to.
    """
    # get alphas from alphas_bar
    alphas = alphas_bar[1:] / alphas_bar[:-1]

    # clip alphas
    alphas = np.clip(alphas, a_min=clip_min, a_max=clip_max, dtype=np.float64)

    # recompute alphas_bar
    alphas_bar = np.cumprod(alphas, axis=0)

    return alphas_bar


def polynomial_decay(
    timesteps: int, s: float = 1e-5, clip_value: float = 0.001, power: float = 2.0
) -> np.ndarray:
    """
    A noise schedule based on a simple polynomial equation from
    https://arxiv.org/abs/2203.17003 to approximate the cosine schedule.

    Args:
        timesteps: number of timesteps T.
        s: precision parameter.
        clip_value: minimum value to clip to.
        power: power of the polynomial.
    """
    # compute alphas_bar
    t = np.linspace(0.0, 1.0, timesteps + 1, dtype=np.float64)
    alphas_bar = (1 - np.power(t, power)) ** 2

    # clip for more stable noise schedule
    alphas_bar = clip_noise_schedule(alphas_bar, clip_min=clip_value)

    # add precision
    precision = 1 - 2 * s
    alphas_bar = precision * alphas_bar + s

    return alphas_bar


def cosine_decay(
    timesteps, s: float = 0.008, v: float = 1.0, clip_value: float = 0.001
) -> np.ndarray:
    """
    Cosine schedule with clipping from https://arxiv.org/abs/2102.09672.

    Args:
        timesteps: number of timesteps T.
        s: precision parameter.
        v: decay parameter.
        clip_value: minimum value to clip to.
    """
    # compute alphas_bar
    t = np.linspace(0.0, 1.0, timesteps + 1, dtype=np.float64)
    f_t = np.cos((t**v + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_bar = f_t / f_t[0]

    # clip for more stable noise schedule
    alphas_bar = clip_noise_schedule(alphas_bar, clip_min=clip_value)

    return alphas_bar


def linear_decay(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> np.ndarray:
    """
    Linear schedule from https://arxiv.org/pdf/2006.11239.pdf.

    Args:
        timesteps: number of timesteps T.
        beta_start: starting value of beta_t, i.e. t=0.
        beta_end: ending value of beta_t, i.e. t=T.
    """
    # rescale beta_start and beta_end as the paper uses 1000 timesteps
    scale = 1000 / timesteps
    beta_start = scale * beta_start
    beta_end = scale * beta_end

    # compute alphas_bar
    betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)

    return alphas_bar


class NoiseSchedule(nn.Module):
    """
    Base class for noise schedules. To be used together with Markovian processes,
    i.e. inheriting from ``MarkovianDiffusion``.
    """

    def __init__(
        self,
        T: int,
        alphas_bar: Union[np.ndarray, list],
        variance_type: str = "lower_bound",
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            T: number of timesteps.
            alphas_bar: noise schedule.
            clip_value: minimum value to clip to.
            variance_type: use either 'lower_bound' or the 'upper_bound'.
            dtype: torch dtype to use for computation accuracy.
        """
        super().__init__()

        self.T = T
        self.alphas_bar = torch.tensor(alphas_bar, dtype=torch.float64, device="cpu")
        self.variance_type = variance_type
        self.dtype = dtype

        if isinstance(self.dtype, str):
            if self.dtype == "float64":
                self.dtype = torch.float64
            elif self.dtype == "float32":
                self.dtype = torch.float32
            else:
                raise ValueError(
                    f"data type must be float32 or float64, got {self.dtype}"
                )

        if self.variance_type == "upper_bound":
            logger.warning(
                "The upper bound for the posterior variance is not the exact one. "
                "This may affect the NLL estimation if used."
            )

        # pre-compute the parameters using double precision
        self.pre_compute_statistics()

    def pre_compute_statistics(self):
        """
        Pre-compute the noise parameters based on the notation of Ho et al.
        """

        self.betas_bar = 1 - self.alphas_bar
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_betas_bar = torch.sqrt(
            self.betas_bar
        )  # different from 1-sqrt(alphas_bar) !

        # infer the different statistics
        self.alphas = self.alphas_bar[1:] / self.alphas_bar[:-1]
        self.alphas = torch.concatenate([self.alphas_bar[:1], self.alphas])
        self.betas = 1.0 - self.alphas
        self.betas_square = self.betas**2
        self.sqrt_betas = torch.sqrt(self.betas)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.inv_sqrt_alphas = 1.0 / self.sqrt_alphas
        self.inv_sqrt_betas_bar = 1.0 / self.sqrt_betas_bar

        # Weither to use the true posterior variance which is the lower bound
        # or the upper bound formulation
        if self.variance_type == "lower_bound":
            self.sigmas_square = self.betas[1:] * (
                self.betas_bar[:-1] / self.betas_bar[1:]
            )

            # lower bound sigma_1 = 0 because beta_bar_0 = 1 - alpha_bar_0 = 1 - 1 = 0
            # We clip to avoid inf when computing decoder loglikelihood or vlb weights
            self.sigmas_square = torch.concatenate(
                [self.sigmas_square[:1], self.sigmas_square]
            )

        elif self.variance_type == "upper_bound":
            self.sigmas_square = self.betas.clone()

            # we always replace the first value by the true posterior variacne
            # to have a better likelihood of L_0
            # see https://arxiv.org/abs/2102.09672
            self.sigmas_square[0] = self.betas[1] * (
                self.betas_bar[0] / self.betas_bar[1]
            )

        else:
            raise ValueError(
                "variance_type must be either 'lower_bound' or 'upper_bound'"
            )

        self.sigmas = torch.sqrt(self.sigmas_square)

        self.vlb_weights = self.betas**2 / (
            2 * self.sigmas_square * self.alphas * self.betas_bar
        )

    def normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the time t to [0, 1].

        Args:
            t: time steps.
        """
        if (t < 0).any() or (t >= self.T).any():
            raise ValueError(
                "t must be between 0 and T-1. This may be due to rounding errors. "
                "The indexing of the noise schedule starts with alpha_bar_1 at idnex 0."
            )

        return t.float() / (self.T - 1)

    def unnormalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Un-normalizes the time t to [0, T-1].

        Args:
            t: normalized time steps.
        """
        if (t < 0.0).any() or (t > 1.0).any():
            raise ValueError("t must be flaot between 0 and 1.")

        return torch.round(t.to(torch.double) * (self.T - 1)).long()

    def forward(self, t: torch.Tensor, keys: list = []) -> Dict[str, torch.Tensor]:
        """
        Query the noise parameters at timestep t.

        Args:
            t: the query timestep.
        """
        if not isinstance(t, torch.Tensor):
            raise ValueError("t must be a torch.Tensor.")

        device = t.device

        if len(t.shape) == 0:
            t = t.reshape(1)

        # convert to integer and numpy
        if t.dtype in [torch.float, torch.double]:
            t = self.unnormalize_time(t)

        t = t.to("cpu")

        # check if out of bounds
        if torch.any(t < 0) or torch.any(t >= self.T):
            raise ValueError(
                "t must be between 0 and T-1. This may be due to rounding errors. "
                "The indexing of the noise schedule starts with alpha_bar_1 at idnex 0."
            )

        # query the noise parameters
        key_to_attr_mapping = {
            "alpha_bar": self.alphas_bar[t],
            "beta_bar": self.betas_bar[t],
            "sqrt_alpha_bar": self.sqrt_alphas_bar[t],
            "sqrt_beta_bar": self.sqrt_betas_bar[t],
            "alpha": self.alphas[t],
            "beta": self.betas[t],
            "sqrt_alpha": self.sqrt_alphas[t],
            "sqrt_beta": self.sqrt_betas[t],
            "beta_square": self.betas_square[t],
            "sigma_square": self.sigmas_square[t],
            "sigma": self.sigmas[t],
            "inv_sqrt_alpha": self.inv_sqrt_alphas[t],
            "inv_sqrt_beta_bar": self.inv_sqrt_betas_bar[t],
            "vlb_weight": self.vlb_weights[t],
        }

        # return all keys if not specified
        if not keys:
            keys = key_to_attr_mapping.keys()  # type: ignore

        # fetch parameters and convert to torch tensors
        params = {}
        for key in keys:
            if key in key_to_attr_mapping:
                params[key] = key_to_attr_mapping[key].to(device=device).to(self.dtype)
            else:
                raise KeyError(
                    f"Key {key} not recognized. "
                    "If using continuous schedule, only few keys are available"
                    " because of the need of the next timestep for the others."
                )

        return params


class CosineSchedule(NoiseSchedule):
    """
    Cosine noise schedule.
    Subclasses ``NoiseSchedule``.
    """

    def __init__(
        self,
        T: int = 1000,
        s: float = 0.008,
        v: float = 1.0,
        clip_value: float = 0.001,
        variance_type: str = "lower_bound",
        **kwargs,
    ):
        """
        Args:
            T: number of steps.
            s: precision parameter.
            v: decay parameter.
            clip_value: clip parameetrs for numerical stability.
            variance_type: use either 'lower_bound' or the 'upper_bound'.
            kwargs: additional keyword arguments.
        """
        self.s = s
        self.v = v
        self.clip_value = clip_value
        alphas_bar = cosine_decay(T, s=self.s, v=self.v, clip_value=self.clip_value)

        super().__init__(
            T,
            alphas_bar,
            variance_type=variance_type,
            **kwargs,
        )


class PolynomialSchedule(NoiseSchedule):
    """
    Polynomial noise schedule.
    Subclasses ``NoiseSchedule``.
    """

    def __init__(
        self,
        T: int = 1000,
        s: float = 1e-5,
        clip_value: float = 0.001,
        variance_type: str = "lower_bound",
        **kwargs,
    ):
        """
        Args:
            T: number of steps.
            s: precision parameter.
            clip_value: clip parameetrs for numerical stability.
            variance_type: use either 'lower_bound' or the 'upper_bound'.
            kwargs: additional keyword arguments.
        """
        self.s = s
        self.clip_value = clip_value
        alphas_bar = polynomial_decay(T, s=s, clip_value=clip_value)

        super().__init__(
            T,
            alphas_bar,
            variance_type=variance_type,
            **kwargs,
        )


class LinearSchedule(NoiseSchedule):
    """
    Linear noise schedule.
    Subclasses ``NoiseSchedule``.
    """

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        variance_type: str = "lower_bound",
        **kwargs,
    ):
        """
        Args:
            T: number of steps.
            beta_start: starting value of beta_t, i.e. t=0.
            beta_end: ending value of beta_t, i.e. t=T.
            variance_type: use either 'lower_bound' or the 'upper_bound'.
            kwargs: additional keyword arguments.
        """
        self.beta_start = beta_start
        self.beta_end = beta_end
        alphas_bar = linear_decay(T, beta_start=beta_start, beta_end=beta_end)

        super().__init__(
            T,
            alphas_bar,
            variance_type=variance_type,
            **kwargs,
        )
