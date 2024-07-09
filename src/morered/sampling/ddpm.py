from typing import Dict, List, Tuple, Union, Optional

import torch
from schnetpack import properties
from torch import nn
from tqdm import tqdm

from morered.processes import DiffusionProcess
from morered.sampling import Sampler
from morered.utils import compute_neighbors, scatter_mean

__all__ = ["DDPM"]


class DDPM(Sampler):
    """
    Implements the plain DDPM ancestral sampler proposed by Ho et al. 2020
    Subclasses the base class 'Sampler'.
    """

    def __init__(
        self,
        diffusion_process: DiffusionProcess,
        denoiser: Union[str, nn.Module],
        time_key: str = "t",
        noise_pred_key: str = "eps_pred",
        **kwargs,
    ):
        """
        Args:
            diffusion_process: The diffusion processe to sample the target property.
            denoiser: denoiser or path to denoiser to use for the reverse process.
            time_key: the key for the time.
            noise_pred_key: the key for the noise prediction.
        """
        super().__init__(diffusion_process, denoiser, **kwargs)
        self.time_key = time_key
        self.noise_pred_key = noise_pred_key

    @torch.no_grad()
    def inference_step(
        self, inputs: Dict[str, torch.Tensor], iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inference step for the model to get the time steps and noise prediction.

        Args:
            inputs: input data for noise prediction.
            iter: the current iteration of the reverse process.
        """
        # current reverse time step
        time_steps = torch.full_like(
            inputs[properties.n_atoms],
            fill_value=iter,
            dtype=torch.long,
            device=self.device,
        )

        # append the normalized time step to the model input
        inputs[self.time_key] = self.diffusion_process.normalize_time(time_steps)

        # broadcast the time step to atoms-level
        inputs[self.time_key] = inputs[self.time_key][inputs[properties.idx_m]]

        # cast input to float for the denoiser
        for key, val in inputs.items():
            if val.dtype == torch.float64:
                inputs[key] = val.float()

        # forward pass through the denoiser
        model_out = self.denoiser(inputs)

        # fetch the noise prediction
        noise_pred = model_out[self.noise_pred_key].detach()

        return time_steps, noise_pred

    @torch.no_grad()
    def denoise(
        self,
        inputs: Dict[str, torch.Tensor],
        t: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Peforns denoising/sampling using the reverse diffusion process.
        Returns the denoised/sampled data, the number of steps taken,
        and the progress history if saved.

        Args:
            inputs: dict with input data in the SchNetPack form,
                    inluding the starting x_t.
            t: the start step of the reverse process, between 1 and T.
                If None, set t = T.
        """
        # Default is t=T
        if t is None:
            t = self.diffusion_process.get_T()

        if not isinstance(t, int) or t < 1 or t > self.diffusion_process.get_T():
            raise ValueError(
                "t must be one int between 1 and T that indicates the starting step."
                "Sampling using different starting steps is not supported yet for DDPM."
            )

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

        # history of the reverse steps
        hist = []

        # simulate the reverse process
        for i in tqdm(range(t - 1, -1, -1)):
            # update the neighbors list if required
            if self.recompute_neighbors:
                batch = compute_neighbors(batch, cutoff=self.cutoff, device=self.device)

            # get the time steps and noise predictions from the denoiser
            time_steps, noise = self.inference_step(batch, i)

            # save history if required. Must be done before the reverse step.
            if self.save_progress and (i % self.progress_stride == 0):
                hist.append(
                    {
                        properties.R: batch[properties.R].cpu().float().clone(),
                        self.time_key: time_steps.cpu(),
                    }
                )

            # perform one reverse step
            batch[properties.R] = self.diffusion_process.reverse_step(
                batch[properties.R],
                noise,
                batch[properties.idx_m],
                time_steps[inputs[properties.idx_m]],
            )

        # prepare the final output
        x_0 = {
            properties.R: (
                batch[properties.R].cpu()
                if self.results_on_cpu
                else batch[properties.R]
            )
        }

        num_steps = torch.full_like(
            batch[properties.n_atoms], t, dtype=torch.long, device="cpu"
        )

        return x_0, num_steps, hist
