from typing import Callable, Optional, Union

import numpy as np
import torch


def mean_squared_error(
    pred: torch.Tensor, target: torch.Tensor, agg_method: Optional[Callable] = None
) -> torch.Tensor:
    """
    Computes the mean squared error.

    Args:
        pred: the predicted values.
        target: the target values.
        agg_method: the aggregation method.
    """
    mse = (pred - target) ** 2
    if agg_method is not None:
        mse = agg_method(mse)
    return mse


def decoder_gaussian_contant(n_dims: Union[torch.Tensor, int], sigma: float) -> float:
    """
    Computes the Gaussian normalization term.

    Args:
        n_dims: the number of dimensions.
        sigma: the variance.
    """
    return -n_dims * (np.log(sigma) + 0.5 * np.log(2 * np.pi))


def decoder_gaussian_nll(
    target_noise: torch.Tensor,
    pred_noise: torch.Tensor,
    sigma_1: float,
    alpha_1: float,
    beta_1: float,
    n_dims: Optional[Union[torch.Tensor, int]] = None,
) -> torch.Tensor:
    """
    Computes the Gaussian NLL of the decoder (i.e. L_0) for a each molecule.

    Args:
        target_noise: ground truth noise.
        pred_noise: predicted noise.
        sigma_1: the variance of the reverse kernel at t.
        alpha_1: the signal portion of the forward kernel at t.
        beta_1: the noise portion of the marginal at t.
        n_dims: the number of dimensions.
    """
    if n_dims is None:
        n_dims = pred_noise.shape[0]

    # get the Gaussian normalization term
    decoder_constant = decoder_gaussian_contant(n_dims, sigma_1**0.5)

    # get the time dependent loss weighting term
    decoder_weighting_term = -0.5 * beta_1 / (sigma_1 * alpha_1)

    # compute the noise loss
    noise_mse = mean_squared_error(pred_noise, target_noise)
    noise_mse = noise_mse.sum(dim=-1)

    # compute the Gaussian NLL
    l_0 = decoder_weighting_term * noise_mse
    l_0 += decoder_constant
    l_0 *= -1.0

    return l_0


def l_t_kl(
    target_noise: torch.Tensor,
    pred_noise: torch.Tensor,
    beta_square: torch.Tensor,
    sigma_square: torch.Tensor,
    alpha: torch.Tensor,
    beta_bar: torch.Tensor,
    weighted: bool = True,
) -> torch.Tensor:
    """
    Computes the L_t loss term for each sampled t.

    Args:
        target_noise: ground truth noise.
        pred_noise: predicted noise.
        beta_square: the variance of the forward kernel at t.
        sigma_square: the variance of the reverse kernel at t.
        alpha: the signal portion of the forward kernel at t.
        beta_bar: the noise portion of the marginal at t.
        weighted: whether to use the VLB time dependent weighting term.
    """
    # Use the VLB time dependent weighting term
    if weighted:
        # convert them to numpy float64 values for rounding error
        beta_square = beta_square.cpu().numpy().astype(np.float64)
        sigma_square = sigma_square.cpu().numpy().astype(np.float64)
        alpha = alpha.cpu().numpy().astype(np.float64)
        beta_bar = beta_bar.cpu().numpy().astype(np.float64)

        weighting_term = (
            torch.from_numpy(beta_square / (sigma_square * alpha * beta_bar))
            .float()
            .to(target_noise.device)
        )
    # Use the time independent weighting term introduced by Ho et al.
    else:
        weighting_term = torch.tensor(1.0, device=target_noise.device)

    # compute the noise loss
    noise_mse = mean_squared_error(pred_noise, target_noise)
    noise_mse = noise_mse.sum(dim=-1)

    # compute the L_t loss term for each sampled t
    l_t = 0.5 * weighting_term * noise_mse

    return l_t


def kl_gaussian(
    mu_q: torch.Tensor,
    var_q: torch.Tensor,
    mu_p: Optional[torch.Tensor] = None,
    var_p: Optional[torch.Tensor] = None,
    n_dims: Optional[Union[torch.Tensor, int]] = None,
) -> torch.Tensor:
    """
    Computes the KL divergence between two isotropic Gaussian distributions.

    Args:
        mu_q: The mean of the first Gaussian.
        var_q: The variance of the first Gaussian.
        mu_p: The mean of the second Gaussian. If None, set to zero.
        var_p: The variance of the second Gaussian. If None, set to one.
        n_dims: The Gaussian dimensionality of each input sample.
    """
    if n_dims is None:
        n_dims = mu_q.shape[0]

    # set prior to standard normal if not provided
    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    if var_p is None:
        var_p = torch.ones_like(var_q)

    # compute the KL divergence
    mu_norm_squeared = mean_squared_error(mu_q, mu_p)
    mu_norm_squeared = mu_norm_squeared.sum(dim=-1)
    return (
        n_dims * 0.5 * (torch.log(var_p) - torch.log(var_q))
        + (n_dims * var_q + mu_norm_squeared) / (2 * var_p)
        - 0.5 * n_dims
    )


def prior_l_T_kl(
    x_0: torch.Tensor, sqrt_alpha_bar_T: float, beta_bar_T: float, n_dims: torch.Tensor
) -> torch.Tensor:
    """
    Computes the KL divergence between the fixed prior isotropic Gaussian
    and the latent diffusion posterior at time T.

    Args:
        x_0: initial atom positions before diffusion.
        sqrt_alpha_bar_T: signal proportion at time T.
        sqrt_beta_bar_T: noise proportion at time T.
        n_dims: The Gaussian dimensionality of each input sample.
    """
    # compute the Gaussian parameters of the posterior at time T
    mu_q_T = sqrt_alpha_bar_T * x_0
    var_q_T = torch.full_like(mu_q_T[:, 0], beta_bar_T)

    return kl_gaussian(mu_q_T, var_q_T, n_dims=n_dims)
