from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from schnetpack import properties
from torch import nn
from tqdm import tqdm

from morered.processes import DiffusionProcess
from morered.sampling import Sampler
from morered.utils import compute_neighbors, scatter_mean

__all__ = ["MoreRed", "MoreRedJT", "MoreRedITP", "MoreRedAS"]


class MoreRed(Sampler):
    """
    Base class for MoreRed samplers as proposed by Kahouli et al. 2024
    """

    def __init__(
        self,
        diffusion_process: DiffusionProcess,
        denoiser: Union[str, nn.Module],
        time_key: str = "t",
        noise_pred_key: str = "eps_pred",
        time_pred_key: str = "t_pred",
        convergence_step: int = 0,
        **kwargs,
    ):
        """
        Args:
            diffusion_process: The diffusion processe to sample the target property.
            denoiser: denoiser or path to denoiser to use for the reverse process.
            time_key: the key for the time.
            noise_pred_key: the key for the noise prediction.
            time_pred_key: the key for the time prediction.
            convergence_step: The time step at which the reverse process converge.
        """
        super().__init__(diffusion_process, denoiser, **kwargs)
        self.time_key = time_key
        self.noise_pred_key = noise_pred_key
        self.time_pred_key = time_pred_key
        self.convergence_step = convergence_step

    @abstractmethod
    def inference_step(
        self, inputs: Dict[str, torch.Tensor], iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inference step for the model to get one time step per molecule
        and the noise prediction.

        Args:
            inputs: input data for noise prediction.
            iter: the current iteration of the reverse process.
        """
        raise NotImplementedError

    @torch.no_grad()
    def denoise(
        self,
        inputs: Dict[str, torch.Tensor],
        max_steps: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Peforns denoising/sampling using the adaptive reverse diffusion process.
        Returns the denoised/sampled data, the number of steps taken,
        and the progress history if saved.

        Args:
            inputs: dict with input data in the SchNetPack form,
                    inluding the starting x_t.
            max_steps: the maximum number of reverse steps to perform.
        """
        # Default maximum number of steps
        if max_steps is None:
            max_steps = self.diffusion_process.get_T()

        # copy inputs to avoid inplace operations
        batch = {prop: val.clone().to(self.device) for prop, val in inputs.items()}

        # check if center of geometry is close to zero
        CoG = scatter_mean(
            batch[properties.R], batch[properties.idx_m], batch[properties.n_atoms]
        )
        if self.diffusion_process.invariant and (CoG > 1e-5).any():
            raise ValueError(
                "The input positions are not centered, "
                "while the specified diffusion process is invariant."
            )

        # set all atoms as neighbors and compute neighbors only once before starting.
        if not self.recompute_neighbors:
            batch = compute_neighbors(batch, fully_connected=True, device=self.device)

        # initialize convergence flag for each molecule
        converged = torch.zeros_like(
            batch[properties.n_atoms], dtype=torch.bool, device=self.device
        )

        # initialize the number of steps taken for each molecule
        num_steps = torch.full_like(
            batch[properties.n_atoms], -1, dtype=torch.long, device=self.device
        )

        # history of the reverse steps
        hist = []

        # simulate the reverse process
        iter = 0
        pbar = tqdm()
        while iter < max_steps:
            # update the neighbors list if required
            if self.recompute_neighbors:
                batch = compute_neighbors(batch, cutoff=self.cutoff, device=self.device)

            # get the time steps and noise predictions from the denoiser
            time_steps, noise = self.inference_step(batch, iter)

            # save history if required. Must be done before the reverse step.
            if self.save_progress and (
                iter % self.progress_stride == 0 or iter == max_steps - 1
            ):
                hist.append(
                    {
                        properties.R: batch[properties.R].cpu().float().clone(),
                        self.time_key: time_steps.cpu(),
                    }
                )

            # perform one reverse step
            x_t = batch[properties.R]
            x_t_1 = self.diffusion_process.reverse_step(
                x_t,
                noise,
                batch[properties.idx_m],
                time_steps[inputs[properties.idx_m]],
            )

            # update only non-converged molecules.
            mask_converged = converged[batch[properties.idx_m]]
            batch[properties.R] = (
                mask_converged.unsqueeze(-1) * x_t
                + (~mask_converged).unsqueeze(-1) * x_t_1
            )

            # use the average time step for convergence check
            converged = converged | (time_steps <= self.convergence_step)

            iter += 1
            pbar.update(1)

            # save the number of steps
            num_steps[converged & (num_steps < 0)] = iter

            # check if all molecules converged and end the denoising
            if converged.all():
                break

        pbar.close()

        # prepare the final output
        x_0 = {
            properties.R: (
                batch[properties.R].cpu()
                if self.results_on_cpu
                else batch[properties.R]
            )
        }

        num_steps = num_steps.cpu()

        return x_0, num_steps, hist


class MoreRedJT(MoreRed):
    """
    Implements the adaptive MoreRed-JT sampler/denoiser proposed by Kahouli et al. 2024
    """

    @torch.no_grad()
    def inference_step(
        self, inputs: Dict[str, torch.Tensor], iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inference step for the model to get one time step per molecule
        and the noise prediction.

        Args:
            inputs: input data for noise prediction.
            iter: the current iteration of the reverse process.
        """
        # cast input to float for the denoiser
        for key, val in inputs.items():
            if val.dtype == torch.float64:
                inputs[key] = val.float()

        # forward pass through the denoiser
        model_out = self.denoiser(inputs)

        # fetch the noise prediction
        noise_pred = model_out[self.noise_pred_key].detach()

        # current reverse time step prediction
        time_steps = model_out[self.time_pred_key].detach()

        # clip the time steps to [0, 1] for outlier predictions
        time_steps = torch.clamp(time_steps, min=0.0, max=1.0)

        # average the time steps at molecule level
        # Note that moving atoms with different time steps breaks the invariance!
        time_steps = scatter_mean(
            time_steps, inputs[properties.idx_m], inputs[properties.n_atoms]
        )

        # map to int in 0, T-1
        time_steps = self.diffusion_process.unnormalize_time(time_steps)

        return time_steps, noise_pred


class MoreRedAS(MoreRed):
    """
    Implements the adaptive MoreRed-AS sampler/denoiser proposed by Kahouli et al. 2024
    """

    def __init__(
        self,
        diffusion_process: DiffusionProcess,
        denoiser: Union[str, nn.Module],
        time_predictor: Union[str, nn.Module],
        **kwargs,
    ):
        """
        Args:
            diffusion_process: The diffusion processe to sample the target property.
            denoiser: denoiser or path to denoiser to use for the reverse process.
            time_predictor: Seperate diffusion time step predictor or path to the model.
                            Used for 'MoreRed-ITP' and 'MoreRed-AS'.
        """
        super().__init__(diffusion_process, denoiser, **kwargs)

        self.time_predictor = time_predictor

        if isinstance(self.time_predictor, str):
            self.time_predictor = torch.load(
                self.time_predictor, device=self.device
            ).eval()
        else:
            self.time_predictor = self.time_predictor.to(self.device).eval()

    @torch.no_grad()
    def get_time_steps(
        self, inputs: Dict[str, torch.Tensor], iter: int
    ) -> torch.Tensor:
        """
        Calculates the current time steps for the reverse process.

        Args:
            inputs: input data for the time predictor.
            iter: the current iteration of the reverse process.
        """
        # cast input to float for the time predictor
        for key, val in inputs.items():
            if val.dtype == torch.float64:
                inputs[key] = val.float()

        # predict the time steps
        time_steps = self.time_predictor(inputs)[self.time_pred_key].detach()

        # clip the time steps to [0, 1] for outlier predictions
        time_steps = torch.clamp(time_steps, min=0.0, max=1.0)

        # average the time steps at molecule level
        # Note that moving atoms with different time steps breaks the invariance!
        time_steps = scatter_mean(
            time_steps, inputs[properties.idx_m], inputs[properties.n_atoms]
        )

        # map to int in 0, T-1
        time_steps = self.diffusion_process.unnormalize_time(time_steps)

        return time_steps

    @torch.no_grad()
    def inference_step(
        self, inputs: Dict[str, torch.Tensor], iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inference step for the model to get one time step per molecule
        and the noise prediction.

        Args:
            inputs: input data for noise prediction.
            iter: the current iteration of the reverse process.
        """
        # get the current time steps
        time_steps = self.get_time_steps(inputs, iter)

        # append the normalized time step to the model input
        # We first unnormlize the time steps to get a binned step as during training
        inputs[self.time_key] = self.diffusion_process.normalize_time(time_steps)
        inputs[self.time_key] = inputs[self.time_key][inputs[properties.idx_m]]

        # cast input to float for the denoiser
        for key, val in inputs.items():
            if val.dtype == torch.float64:
                inputs[key] = val.float()

        # fetch the noise prediction
        model_out = self.denoiser(inputs)
        noise_pred = model_out[self.noise_pred_key].detach()

        return time_steps, noise_pred


class MoreRedITP(MoreRedAS):
    """
    Implements the MoreRed-ITP sampler/denoiser proposed by Kahouli et al. 2024
    """

    @torch.no_grad()
    def get_time_steps(
        self, inputs: Dict[str, torch.Tensor], iter: int
    ) -> torch.Tensor:
        """
        Calculates the current time steps for the reverse process.

        Args:
            inputs: input data for the time predictor.
            iter: the current iteration of the reverse process.
        """
        if iter == 0:
            # if initial iteration, predict the time steps using the time predictor
            time_steps = super().get_time_steps(inputs, iter)

            # persist the initial time steps
            self._init_time_steps = time_steps.clone()

        else:
            # get the current time steps from the predicted initial time steps
            time_steps = torch.clamp(self._init_time_steps - iter, min=0)

        return time_steps
