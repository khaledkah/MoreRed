from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch

from morered.noise_schedules import NoiseSchedule
from morered.processes.base import DiffusionProcess
from morered.utils import batch_center_systems
from morered.processes.functional import _check_shapes, sample_isotropic_Gaussian

__all__ = ["GaussianDDPM", "VPGaussianDDPM"]


class GaussianDDPM(DiffusionProcess):
    """
    Base class for DDPM models using Gaussian diffusion kernels.
    """

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        noise_key: str = "eps",
        **kwargs,
    ):
        """
        Args:
            noise_schedule: noise schedule for the diffusion process.
            noise_key: key to store the Gaussian noise.
            kwargs: additional arguments to be passed to parent class.
        """
        super().__init__(**kwargs)
        self.noise_schedule = noise_schedule
        self.noise_key = noise_key

    @abstractmethod
    def perturbation_kernel(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        get the mean and std of the Gaussian perturbation kernel
        p(x_t|x_0) = N(mean(x_0,t),std(t)).

        Args:
            x_0: input tensor x_0 ~ p_data(x_0) to be diffused.
            t: time step.
        """
        raise NotImplementedError

    @abstractmethod
    def transition_kernel(
        self, x_t: torch.Tensor, t_next: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        get the mean and std of the transition kernel of the Gaussian Markov process,
        i.e. p(x_t+1|x_t) = N(mean(x_t,t+1),std(t+1)).

        Args:
            x_t: input tensor x_t ~ p_t at step t.
            t_next: next time step t+1.
            kwargs: additional keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def prior(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the Gaussian prior distribution p(x_T) = N(mean,std).

        Args:
            x: dummy input tensor, e.g. to infer shape.
            **kargs: additional keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def reverse_kernel(
        self,
        x_t: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the reverse transition kernel p(x_t-1|x_t)=N(mean(x_t,t),std(t))
        of the Markov process.

        Args:
            x_t: input tensor x_t ~ p_t at step t.
            noise: the (predicted) Gaussian noise to be removed.
            t: current time steps.
        """
        raise NotImplementedError

    def get_T(self) -> int:
        """
        Returns the total number of diffusion steps T.
        """
        return self.noise_schedule.T

    def normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the time t to [0, 1].

        Args:
            t: time steps as integer in [0, T-1].
        """
        return self.noise_schedule.normalize_time(t)

    def unnormalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Un-normalizes the time t to [0, T-1].

        Args:
            t: normalized time steps as float in [0, 1].
        """
        return self.noise_schedule.unnormalize_time(t)

    def forward_step(
        self,
        x_t: torch.Tensor,
        idx_m: Optional[torch.Tensor],
        t_next: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one Markov transition step to sample x_t+1 ~ p(x_t+1|x_t).

        Args:
            x_t: input tensor x_t ~ p_t at diffusion step t.
            idx_m: same as ``proporties.idx_m`` to assign each row of x to its system.
                    Set to None if one system or no invariance needed.
            t_next: next time steps t+1.
            kwargs: additional keyword arguments.
        """
        # convert to correct dtype
        x_t = x_t.to(self.dtype)

        x_t, t_next = _check_shapes(x_t, t_next)

        # get the mean and std of the transition kernel.
        mean, std = self.transition_kernel(x_t, t_next, **kwargs)

        # sample x_t+1.
        x_next, noise = sample_isotropic_Gaussian(
            mean, std, invariant=self.invariant, idx_m=idx_m, **kwargs
        )

        return x_next, noise

    def diffuse(
        self,
        x_0: torch.Tensor,
        idx_m: Optional[torch.Tensor],
        t: torch.Tensor,
        output_key: str = "x_t",
        return_dict: bool = False,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Diffuses origin x_0 by t steps to sample x_t from p(x_t|x_0),
        given x_0 ~ p_data. Return tuple of tensors x_t and noise,
        or Dict of tensors with x_t and other quantities of interest.

        Args:
            x_0: input tensor x_0 ~ p_data to be diffused.
            idx_m: same as ``proporties.idx_m`` to assign each row to its system.
                Set to None if one system or no invariance needed.
            t: time steps.
            output_key: key to store the diffused x_t.
            return_dict: if True, return results under a dictionary of tensors.
            kwargs: additional keyword arguments.
        """
        # convert to correct data type.
        x_0 = x_0.to(self.dtype)

        x_0, t = _check_shapes(x_0, t)

        # query noise parameters.
        mean, std = self.perturbation_kernel(x_0, t)

        # sample by Gaussian diffusion.
        x_t, noise = sample_isotropic_Gaussian(
            mean, std, invariant=self.invariant, idx_m=idx_m, **kwargs
        )

        if return_dict:
            return {output_key: x_t, self.noise_key: noise}
        else:
            return x_t, noise

    def sample_prior(
        self, x: torch.Tensor, idx_m: Optional[torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """
        Samples from the prior distribution p(x_T) = N(mean,std).

        Args:
            x: dummy input tensor, e.g. to infer shape.
            idx_m: same as ``proporties.idx_m`` to assign each row to its system.
            **kargs: additional keyword arguments.
        """
        x = x.to(self.dtype)

        # get the mean and std of the prior.
        mean, std = self.prior(x, **kwargs)

        # sample from the prior.
        x_T, _ = sample_isotropic_Gaussian(
            mean, std, invariant=self.invariant, idx_m=idx_m, **kwargs
        )

        return x_T

    def reverse_step(
        self,
        x_t: torch.Tensor,
        model_out: Union[torch.Tensor, Dict[str, torch.Tensor]],
        idx_m: Optional[torch.Tensor],
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Performs one reverse diffusion step to sample x_t-i ~ p(x_t-i|x_t), 0<i.

        Args:
            x_t: input tensor x_t ~ p_t at diffusion step t.
            model_out: output of the denoiser model.
            idx_m: same as ``proporties.idx_m`` to assign each row of x to its system.
                Set to None if one system or no invariance needed.
            t: time steps.
            **kwargs: additional keyword arguments for subclasses.
        """
        # convert to correct dtypes.
        x_t = x_t.to(self.dtype)

        x_t, t = _check_shapes(x_t, t)

        # get the noise prediction from the model output.
        if not isinstance(model_out, torch.Tensor):
            model_out = model_out[self.noise_key]
        noise = model_out.to(self.dtype)

        # if invariant, center the noise to zero center of geometry.
        # Safeguard if model prediction was not centered.
        if self.invariant:
            noise = batch_center_systems(noise, idx_m, None)

        # get the mean and std of the reverse transition kernel.
        mean, std = self.reverse_kernel(x_t, noise, t)

        # sample by Gaussian diffusion.
        x_t, _ = sample_isotropic_Gaussian(
            mean, std, invariant=self.invariant, idx_m=idx_m, **kwargs
        )

        return x_t


class VPGaussianDDPM(GaussianDDPM):
    """
    Variance Preserving DDPM model using Gaussian diffusion kernels.
    As proposed in HO et al. 2020 (https://arxiv.org/abs/2006.11239).
    """

    def __init__(self, noise_schedule: NoiseSchedule, **kwargs):
        """
        Args:
            noise_schedule: noise schedule to use for diffusion.
            kwargs: additional keyword arguments.
        """
        super().__init__(noise_schedule, **kwargs)

    def transition_kernel(
        self, x_t: torch.Tensor, t_next: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the transition kernel
        of the VP Gaussian Markov process,

        Args:
            x_t: input tensor x_t ~ p_t at step t.
            t_next: next time steps t+1.
            kwargs: additional keyword arguments.
        """
        x_t = x_t.to(self.dtype)

        # query noise parameters.
        noise_params = self.noise_schedule(t_next, keys=["sqrt_alpha", "sqrt_beta"])
        sqrt_alpha = noise_params["sqrt_alpha"]
        sqrt_beta = noise_params["sqrt_beta"]

        # get the mean and std of the transition kernel.
        mean = x_t * sqrt_alpha
        std = sqrt_beta

        return mean, std

    def perturbation_kernel(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the Gaussian perturbation kernel
        p(x_t|x_0) = N(mean(x_0,t),std(t)).

        Args:
            x_0: input tensor x_0 ~ p_data(x_0) to be diffused.
            t: time steps.
        """
        x_0 = x_0.to(self.dtype)

        # query noise parameters.
        noise_params = self.noise_schedule(t, keys=["sqrt_alpha_bar", "sqrt_beta_bar"])
        sqrt_alpha_bar = noise_params["sqrt_alpha_bar"]
        sqrt_beta_bar = noise_params["sqrt_beta_bar"]

        # get the mean and std of the Gaussian perturbation kernel.
        mean = x_0 * sqrt_alpha_bar
        std = sqrt_beta_bar

        return mean, std

    def prior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the prior distribution p(x_T) = N(0,I).

        Args:
            x: dummy input tensor, e.g. to infer shape.
        """
        mean = torch.zeros_like(x, dtype=self.dtype)
        std = torch.ones_like(x, dtype=self.dtype)

        return mean, std

    def reverse_kernel(
        self,
        x_t: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the reverse transition kernel p(x_t-1|x_t)=N(mean(x_t-1,t),std(t))
        of the Markov process.

        Args:
            x_t: input tensor x_t ~ p_t at step t.
            noise: Gaussian noise to be reversed.
            t: current time steps.
        """
        # convert to correct dtype
        x_t = x_t.to(self.dtype)
        noise = noise.to(self.dtype)

        # get noise schedule parameters
        noise_params = self.noise_schedule(
            t,
            keys=["inv_sqrt_alpha", "beta", "inv_sqrt_beta_bar", "sigma"],
        )
        inv_sqrt_alpha_t = noise_params["inv_sqrt_alpha"]
        beta_t = noise_params["beta"]
        inv_sqrt_beta_t_bar = noise_params["inv_sqrt_beta_bar"]
        sigma_t = noise_params["sigma"]

        # add noise with variance \sigma (stochasticity) only for t!=0
        # otherwise return the mean \mu as the final sample
        sigma_t *= t != 0

        # compute the mean \mu of the reverse kernel
        mu = inv_sqrt_alpha_t * (x_t - (beta_t * inv_sqrt_beta_t_bar) * noise)

        return mu, sigma_t
