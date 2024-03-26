from typing import Optional, Tuple

import torch

from morered.utils import batch_center_systems


def _check_shapes(
    x: torch.Tensor, t: torch.Tensor, t_next: Optional[torch.Tensor] = None
) -> Tuple:
    """
    Checks and fixes the shapes of x, t and t_next.

    Args:
        x: input tensor.
        t: current time steps.
        t_next: next time steps.
    """
    if t_next is not None and t.shape != t_next.shape:
        raise ValueError(
            "t and t_next must have the same shape. "
            f"Got t.shape={t.shape} and t_next.shape={t_next.shape}."
        )

    if len(x.shape) < len(t.shape):
        x = x.unsqueeze(0).repeat_interleave(t.shape[0], 0)

    while len(x.shape) > len(t.shape):
        t = t.unsqueeze(-1)
        if t_next is not None:
            t_next = t_next.unsqueeze(-1)

    if t_next is None:
        return x, t
    else:
        return x, t, t_next


def sample_noise(
    shape: Tuple,
    invariant: bool,
    idx_m: Optional[torch.Tensor],
    device: torch.device,
    n_atoms: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Sample Gaussian noise based on input shape.
    Project to the zero center of geometry if invariant.

    Args:
        shape: shape of the noise.
        invariant: if True, apply invariance constraint.
        idx_m: same as ``proporties.idx_m`` to map each row of x to its system.
                Set to None if one system or no invariance needed.
        device: torch device to store the noise tensor.
        n_atoms: number of atoms per system.
        dtype: data type to use for computation accuracy.
    """
    # sample noise
    noise = torch.randn(shape, device=device, dtype=dtype)

    # The invariance trick: project noise to the zero center of geometry.
    if invariant:
        # system-wise center of geometry
        if idx_m is not None:
            # infer n_atoms from idx_m if not passed.
            if n_atoms is None:
                _, n_atoms = torch.unique_consecutive(idx_m, return_counts=True)

                if len(n_atoms) != len(torch.unique(idx_m)):  # type: ignore
                    raise ValueError(
                        "idx_m of the same system must be consecutive."
                        " Alternatively, pass n_atoms per system as input."
                    )

            noise = batch_center_systems(noise, idx_m, n_atoms)  # type: ignore

        # global center of geometry if one system passed.
        else:
            noise -= noise.mean(-2).unsqueeze(-2)

    return noise


def sample_noise_like(
    x: torch.Tensor,
    invariant: bool,
    idx_m: Optional[torch.Tensor],
    n_atoms: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample Gaussian noise based on input x.

    Args:
        x: input tensor, e.g. to infer shape.
        invariant: if True, apply invariance constraint.
        idx_m: same as ``proporties.idx_m`` to map each row of x to its system.
                Set to None if one system or no invariance needed.
        n_atoms: number of atoms per system.
    """
    return sample_noise(x.shape, invariant, idx_m, x.device, n_atoms, dtype=x.dtype)


def sample_isotropic_Gaussian(
    mean: torch.Tensor,
    std: torch.Tensor,
    invariant: bool,
    idx_m: Optional[torch.Tensor],
    n_atoms: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
):
    """
    Use the reparametrization trick to Sample from iso Gaussian distribution
    with given mean and std.

    Args:
        mean: mean of the Gaussian distribution.
        std: standard deviation of the Gaussian distribution.
        invariant: if True, the noise is projected to the zero center of geometry.
                 The mean is computed over the -2 dimension.
        idx_m: same as ``proporties.idx_m`` to map each row of x to its system.
                Set to None if one system or no invariance needed.
        n_atoms: number of atoms per system.
        noise: the Gaussian noise. If None, a new noise is sampled.
                Otherwise, ``idx_m``and ``n_atoms`` and ``invariant`` are ignored.
    """
    # sample noise if not given.
    if noise is None:
        noise = sample_noise_like(mean, invariant, idx_m, n_atoms)

    # sample using the Gaussian reparametrization trick.
    sample = mean + std * noise

    return sample, noise
