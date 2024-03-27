from typing import Dict

import torch
from schnetpack import properties
from torchmetrics import Metric

from morered.noise_schedules import NoiseSchedule
from morered.optimization.losses import decoder_gaussian_nll, l_t_kl, prior_l_T_kl


class NLL(Metric):
    """
    Computes the diffusion model's VLB of the NLL: L = L_0 + sum_t(L_t) + L_T.
    Uses the unbiased approxiamtion sum_t(L_t) = T.E(L_i) with i = Dist(1, T-1)
    instead of computing the full sum. Usually Dis(.) = Uniform(.).
    L_0 is computed using the decoder's NLL, which is usually much larger than
    the other terms so it may always be included in the loss.
    """

    full_state_update = False

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        diffuse_property: str,
        include_l0: bool = True,
        include_lT: bool = True,
        training: bool = False,
        time_key: str = "diff_step",
        noise_key: str = "eps",
        noise_pred_key: str = "eps_pred",
        t0_noise_key: str = "eps_init",
        t0_noise_pred_key: str = "eps_init_pred",
    ):
        """
        Args:
            noise_schedule: noise schedule object used for noise parameters.
            diffuse_property: property to diffuse.
            include_l0: whether to always compute the decoder's L_0 term
                        or only when it's sampled from t=Dist(0, T-1).
            include_lT: whether to explicitly compute the L_T term or set to 0.
            training: whether the NLL is used as loss for training.
            time_key: dict key of the time/diffusion step.
            noise_key: dict key of the true noise.
            noise_pred_key: dict key of the predicted noise.
            t0_noise_key: dict key of the initial noise at t=0.
                            Needed only if include_l0=True.
            t0_noise_pred_key: dict key for the predicted initial noise at t=0.
                                Needed only if include_l0=True.
        """
        super().__init__()

        self.noise_schedule = noise_schedule
        self.diffuse_property = diffuse_property
        self.include_l0 = include_l0
        self.include_lT = include_lT
        self.training = training
        self.time_key = time_key
        self.noise_key = noise_key
        self.noise_pred_key = noise_pred_key
        self.t0_noise_key = t0_noise_key
        self.t0_noise_pred_key = t0_noise_pred_key

        self.add_state("l0", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("lt", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("lT", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _compute_l0(
        self, inputs: Dict[str, torch.Tensor], n_dims: torch.Tensor
    ) -> torch.Tensor:
        """
        compute the decoder's NLL term L_0.

        Args:
            inputs: input tensors with noise prediction and ground truth.
            n_dims: The Gaussian dimensionality of each input sample.
        """
        # query the noise parameters for t=0 using numpy float64
        sigma_1 = self.noise_schedule.sigmas[0]
        alpha_1 = self.noise_schedule.alphas[0]
        beta_1 = self.noise_schedule.betas[0]

        # mask for the sampled diffusion step t = 0
        steps = torch.round(inputs[self.time_key] * self.noise_schedule.T)
        l_0_mask = steps == 0

        # Always compute the L_0 term of the loss
        if self.include_l0:
            if self.t0_noise_key not in inputs or self.t0_noise_pred_key not in inputs:
                raise ValueError(
                    f"Keys {self.t0_noise_key} and/or {self.t0_noise_pred_key}"
                    f"not found in the input dict, while 'include_l0' is set to 'True'."
                    f"Check that 'include_l0' is set to 'True' in Task as well."
                )

            if l_0_mask.any() and self.training:
                raise ValueError(
                    "t = 0 can't be included in the sum of L_t if 'include_l0'"
                    "is set to True and vlb is used for training "
                    "(i.e. 'training' is set to True)"
                )

            l_0 = decoder_gaussian_nll(
                inputs[self.t0_noise_key],
                inputs[self.t0_noise_pred_key],
                sigma_1,
                alpha_1,
                beta_1,
                n_dims,
            )

        # Include L_0 only if t=0 is sampled
        else:
            l_0 = decoder_gaussian_nll(
                inputs[self.noise_key],
                inputs[self.noise_pred_key],
                sigma_1,
                alpha_1,
                beta_1,
                n_dims,
            )
            l_0 = l_0 * l_0_mask.float()

        return l_0.mean()

    def _compute_lt(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        computes the NLL term sum_t(L_t) for t=1,...,T-1.

        Args:
            inputs: input tensors with noise prediction and ground truth.
        """
        # query the noise parameters
        noise_params = self.noise_schedule(inputs[self.time_key])
        beta_bar = noise_params["beta_bar"]
        beta_square = noise_params["beta_square"]
        alpha = noise_params["alpha"]
        sigma_square = noise_params["sigma"]

        # compute the KL divergences for L_t
        l_t = l_t_kl(
            inputs[self.noise_key],
            inputs[self.noise_pred_key],
            beta_square,
            sigma_square,
            alpha,
            beta_bar,
        )

        # exlude t=0 from the sum of L_t
        steps = torch.round(inputs[self.time_key] * self.noise_schedule.T)
        l_0_mask = steps == 0
        l_t = l_t * (~l_0_mask).float()

        # unbiased approximation: sum_t(L_t)=T.E(L_t)
        l_t = self.noise_schedule.T * l_t.mean()

        return l_t

    def _compute_lT(
        self, inputs: Dict[str, torch.Tensor], n_dims: torch.Tensor
    ) -> torch.Tensor:
        """
        computes the NLL term L_T for t=T, i.e. the prior.

        Args:
            inputs: input tensors with noise prediction and ground truth.
            n_dims: The Gaussian dimensionality of each input sample.
        """
        # Always explicitly compute the prior term L_T
        if self.include_lT:
            sqrt_alpha_bar_T = self.noise_schedule.sqrt_alphas_bar[-1]
            beta_bar_T = self.noise_schedule.betas_bar[-1]

            if f"original_{self.diffuse_property}" not in inputs:
                raise ValueError(
                    f"required key 'original_{self.diffuse_property}'"
                    "when 'include_LT=True' not found in input dictionary."
                )

            l_T = prior_l_T_kl(
                inputs[f"original_{self.diffuse_property}"],
                sqrt_alpha_bar_T,
                beta_bar_T,
                n_dims,
            )

        # Set L_T to 0 if negligable per definition
        else:
            l_T = torch.tensor([0.0])
        return l_T.mean()

    def update(self, inputs: Dict[str, torch.Tensor]):
        """
        overrites the ``update`` method to compute the different NLL terms.

        Args:
            inputs: input tensors with noise prediction and ground truth.
        """
        # number of degrees of freedom after fixing the center of mass
        n_dims = (inputs[properties.n_atoms] - 1) * 3
        n_dims = n_dims[inputs[properties.idx_m]]

        # compute the different NLL terms
        l_0 = self._compute_l0(inputs, n_dims)
        l_t = self._compute_lt(inputs)
        l_T = self._compute_lT(inputs, n_dims)

        # update states
        self.l0 += l_0
        self.lt += l_t
        self.lT += l_T
        self.n_samples += 1.0

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        overrites the ``compute`` method to return the NLL.
        """
        return {
            "l0": self.l0 / self.n_samples,
            "lt": self.lt / self.n_samples,
            "lT": self.lT / self.n_samples,
            "nll": (self.l0 + self.lt + self.lT) / self.n_samples,
        }
