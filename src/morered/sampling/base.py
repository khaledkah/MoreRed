import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from ase import Atoms
from schnetpack import properties
from schnetpack import transform as trn
from schnetpack.data.loader import _atoms_collate_fn
from torch import nn

from morered.processes import DiffusionProcess

logger = logging.getLogger(__name__)
__all__ = ["Sampler"]


class Sampler:
    """
    Base class for for sampling or denoising using reverse diffusion.
    """

    def __init__(
        self,
        diffusion_process: DiffusionProcess,
        denoiser: Union[nn.Module, str],
        cutoff: float = 5.0,
        recompute_neighbors: bool = False,
        save_progress: bool = False,
        progress_stride: int = 1,
        results_on_cpu: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            diffusion_process: The diffusion processe to sample the target property.
            denoiser: denoiser or path to denoiser to use for the reverse process.
            cutoff: the cutoff radius for the neighbor list if recompute neighbors.
            recompute_neighbors: if True, recompute the neighbor list at each iteration.
                                 Otherwise, set all atoms as neighbors at the beginning.
            save_progress: if True, save the progress of the reverse process.
            progress_stride: the stride for saving the progress.
            results_on_cpu: if True, move the returned results to CPU.
            device: the device to use for denoising.
        """
        self.diffusion_process = diffusion_process
        self.denoiser = denoiser
        self.cutoff = cutoff
        self.save_progress = save_progress
        self.progress_stride = progress_stride
        self.recompute_neighbors = recompute_neighbors
        self.results_on_cpu = results_on_cpu
        self.device = device

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(self.denoiser, str):
            self.denoiser = torch.load(self.denoiser, map_location=self.device).eval()
        elif self.denoiser is not None:
            self.denoiser = self.denoiser.to(self.device).eval()

    def _infer_inputs(
        self, system: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Checks the input data and fills missing data with random values,
        for instance, if only atomic numbers Z but no positions are given
        to sample/denoise from p(R|Z).

        Args:
            system: dict containing one input system.
        """
        # check input format
        if not isinstance(system, dict):
            raise ValueError("Inputs must be dicts.")

        # check if all necessary properties are present in the input
        if properties.Z not in system:
            raise NotImplementedError(
                "Atomic numbers must be provided as input."
                " This sampler models the conditional p(R|Z)."
            )

        # get atomic numbers
        numbers = system[properties.Z]

        # get or initialize positions
        if properties.R not in system:
            positions = torch.randn(len(numbers), 3, device=self.device)
        else:
            positions = system[properties.R]

        return numbers, positions

    def prepare_inputs(
        self,
        inputs: List[Union[torch.Tensor, Dict[str, torch.Tensor], Atoms]],
        transforms: Optional[List[trn.Transform]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepares and converts the inputs in SchNetPack format for the sampler.
        Args:
            inputs: the inputs to be converted. Supports:
                        - one element or list of tensors with the atomic numbers Z
                        - one element or list of dicts of tensors including R and Z
                        - one element or list of ase.Atoms
            transforms: Optional transforms to apply to the inputs.
        """
        # set default transforms
        if transforms is None:
            transforms = [
                trn.CastTo64(),
                trn.SubtractCenterOfGeometry(),
            ]

        # check inputs format
        if (
            isinstance(inputs, torch.Tensor)
            or isinstance(inputs, dict)
            or isinstance(inputs, Atoms)
        ):
            inputs = [inputs]
        elif not isinstance(inputs, list):
            raise ValueError(
                "Inputs must be:"
                "one element or list of tensors with the atomic numbers Z "
                "one element or list of Dict of tensors including R and Z "
                "one element or list of ase.Atoms."
            )

        if isinstance(inputs[0], torch.Tensor):
            inputs = [{properties.Z: inp} for inp in inputs]

        # convert inputs to SchNetPack batch format
        batch = []
        for idx, system in enumerate(inputs):
            if isinstance(system, dict):
                # sanity checks
                numbers, positions = self._infer_inputs(system)

                # convert to ase.Atoms
                mol = Atoms(numbers=numbers, positions=positions)
            else:
                mol = system
                system = {}

            # convert to dict of tensors in SchNetPack format
            system.update(
                {
                    properties.n_atoms: torch.tensor(
                        [mol.get_global_number_of_atoms()]
                    ),
                    properties.Z: torch.from_numpy(mol.get_atomic_numbers()),
                    properties.R: torch.from_numpy(mol.get_positions()),
                    properties.cell: torch.from_numpy(mol.get_cell().array).view(
                        -1, 3, 3
                    ),
                    properties.pbc: torch.from_numpy(mol.get_pbc()).view(-1, 3),
                    properties.idx: torch.tensor([idx]),
                }
            )

            # apply transforms
            for transform in transforms:
                system = transform(system)

            batch.append(system)

        # collate batch in a dict of tensors in SchNetPack format
        batch = _atoms_collate_fn(batch)

        # Move input batch to device
        batch = {p: batch[p].to(self.device) for p in batch}

        return batch

    def update_model(self, model: nn.Module):
        """
        Updates the denoising model.

        Args:
            model: the new denoising model.
        """
        self.denoiser = model

    def sample_prior(
        self,
        inputs: Dict[str, torch.Tensor],
        t: Optional[Union[int, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Samples from the prior p(x_t) for the reverse diffusion process.
        It uses the forward diffusion process to diffuse the input data if t not None,
        otherwise it samples from the tractable prior p(x_T).

        Args:
            inputs: inputs dict with x_0 of each target property if sampling from t < T,
                    or dummy inputs for the target properties if sampling from t = T.
            t: the start time step of the reverse process,
                starting at 0 for diffusion step 1 until T-1.
                pass None if sampling from the latent prior p(x_T).
            **kwargs: additional arguments to pass to the reverse process.
        """
        # get x_0 from inputs
        try:
            x_0 = (
                inputs[f"original_{properties.R}"]
                if f"original_{properties.R}" in inputs
                else inputs[properties.R]
            )
        except KeyError:
            raise KeyError(
                f"Input data must contain the true x_0 property under '{properties.R}' "
                f"or 'original_{properties.R}' to be diffused if sampling from t < T, "
                f"or dummy input to infer shape if sampling from t = T."
            )

        # prior for t < T: diffuse using p(x_t | x_0)
        if t is not None:
            if isinstance(t, int):
                t = torch.tensor(t, device=self.device)
            elif not isinstance(t, torch.Tensor):
                raise ValueError("t must be a torch.Tensor or int when not None.")

            t = t.to(self.device)

            # compute priors using the forward diffusion process
            x_t = self.diffusion_process.diffuse(
                x_0,
                inputs[properties.idx_m],
                t,
                return_dict=True,
                output_key="x_t",
                **kwargs,
            )["x_t"]

            # prior for t = T: sample from latent p(x_T)
        else:
            x_t = self.diffusion_process.sample_prior(
                inputs[properties.R], inputs[properties.idx_m], **kwargs
            )

        outputs = {properties.R: x_t.to(device=self.device)}

        return outputs

    @abstractmethod
    def denoise(
        self,
        inputs: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Peforns denoising/sampling using the reverse diffusion process.
        Returns the denoised/sampled data, the number of steps taken,
        and the progress history if saved.

        Args:
            inputs: dict with input data in the SchNetPack form,
                    inluding the starting x_t.
            kwargs: additional keyword arguments.
        """
        raise NotImplementedError

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """
        Defines the default call method.
        Currently equivalent to calling ``self.denoise``.
        """
        return self.denoise(*args, **kwargs)
