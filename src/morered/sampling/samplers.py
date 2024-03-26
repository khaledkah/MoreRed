from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from schnetpack import properties
from schnetpack import transform as trn
from schnetpack.data.loader import _atoms_collate_fn
from morered.processes.functional import forward_marginal_gaussian
from morered.noise_schedules import NoiseSchedule
from morered.utils import batch_center_systems, scatter_mean
from morered.utils import (
    check_validity,
    compute_neighbors,
    generate_bonds_data,
)
from torch import nn
from tqdm import tqdm

__all__ = ["Sampler", "DDPMSampler", "MoreRedSampler"]


class Sampler:
    """
    Base class for reverse diffusion for sampling or denoising.
    """

    def __init__(
        self,
        denoiser: Union[str, nn.Module],
        noise_schedule: NoiseSchedule,
        target_props: List[str],
        invaraint: bool = True,
        cutoff: float = 5.0,
        save_progress: bool = False,
        progress_stride: int = 1,
        recompute_neighbors: bool = False,
        check_validity: bool = False,
        bonds_data: Optional[Dict[str, np.ndarray]] = None,
        device: Optional[torch.device] = None,
        transforms: Optional[List[trn.Transform]] = None,
        results_on_cpu: bool = True,
    ):
        """
        Args:
            denoiser: denoiser or path to denoiser to use for the reverse process.
            noise_schedule: the noise schedule of the reverse process.
            target_props: the target properties to denoise or sample.
            invaraint: if True, use the invariance trick for atom positions.
            cutoff: the cutoff radius for the neighbor list if recompute neighbors.
            save_progress: if True, save the progress of the reverse process.
            progress_stride: the stride for saving the progress.
            recompute_neighbors: if True, recompute the neighbor list at each iteration.
                                 Otherwise, set all atoms as neighbors at the beginning.
            check_validity: if True, check the validity of the molecules in sampling.
            bonds_data: the bonds data to check the validity of the molecules.
            device: the device to use for PyTorch.
            transforms: the transforms to apply to the inputs in prepare_inputs.
            results_on_cpu: if True, move the results to cpu before returning.
        """
        self.denoiser = denoiser
        self.noise_schedule = noise_schedule
        self.target_props = target_props
        self.invariant = invaraint
        self.cutoff = cutoff
        self.save_progress = save_progress
        self.progress_stride = progress_stride
        self.recompute_neighbors = recompute_neighbors
        self.check_validity = check_validity
        self.results_on_cpu = results_on_cpu

        # set default transforms
        self.transforms = transforms or [
            trn.SubtractCenterOfGeometry(),
            trn.MatScipyNeighborList(cutoff=self.cutoff),
            trn.CastTo32(),
        ]

        # set default device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # check if the sampler supports the target properties
        for prop in self.target_props:
            if prop not in [properties.R]:
                raise NotImplementedError(
                    f"Property '{prop}' is not supported for sampling."
                )

        # get bonds statistics to check heuristically molecule validity
        if self.check_validity and bonds_data is None:
            self.bonds_data = generate_bonds_data()

        if isinstance(self.denoiser, str):
            self.denoiser = torch.load(self.denoiser, map_location=self.device).eval()
        elif self.denoiser is not None:
            self.denoiser = self.denoiser.to(self.device).eval()

    def update_model(self, denoiser: nn.Module):
        """
        Update the denoiser model.

        Args:
            denoiser: the new denoiser model.
        """
        self.denoiser = denoiser.to(self.device).eval()

    def prepare_inputs(
        self,
        inputs: List[Union[Dict[str, torch.Tensor], Atoms]],
        additional_inputs: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepares and converts the inputs for the denoiser.

        Args:
            inputs: the inputs to be converted to the denoiser.
            additional_inputs: Optional additional inputs to append to each molecule.
        """
        # check inputs types
        if type(inputs) is dict or type(inputs) is Atoms:
            inputs = [inputs]
        elif type(inputs) is not list:
            raise ValueError("Inputs must be a list of dicts or ase Atoms.")

        if isinstance(additional_inputs, dict):
            additional_inputs = [additional_inputs]

        if additional_inputs is not None and len(inputs) != len(additional_inputs):
            raise ValueError(
                "len of inputs and additional_inputs must be equal."
                f"Got {len(inputs)} and {len(additional_inputs)}."
            )

        batch = []
        for idx, system in enumerate(inputs):
            if isinstance(system, dict):
                # sanity checks
                if not isinstance(system, dict):
                    raise ValueError("Inputs must be dicts.")

                if properties.Z not in system:
                    raise NotImplementedError(
                        "Atomic numbers must be provided. "
                        "Generation of Z is not implemented yet."
                    )
                else:
                    numbers = system[properties.Z]

                if properties.cell in system or properties.pbc in system:
                    raise NotImplementedError(
                        "Cell and PBC generation are not supported yet."
                    )

                if (
                    properties.R not in system
                    and properties.Z not in system
                    and properties.n_atoms not in system
                ):
                    raise ValueError("at least one of R, Z or n_atoms must be provided")

                # get number of atoms
                if properties.n_atoms not in system:
                    n_atoms = (
                        len(system[properties.R])
                        if properties.R in system
                        else len(system[properties.Z])
                    )
                else:
                    n_atoms = system[properties.n_atoms].item()

                # get or initialize positions
                if properties.R not in system:
                    positions = torch.randn(n_atoms, 3)  # type: ignore
                    if self.invariant:
                        positions = positions - positions.mean(dim=0, keepdim=True)
                else:
                    positions = system[properties.R]

                if not (len(numbers) == len(positions) == n_atoms):
                    raise ValueError("len of R and Z must be equal to n_atoms.")

                # convert to ase.Atoms
                mol = Atoms(numbers=numbers, positions=positions)
            else:
                mol = system
                system = {}

            # convert to dict of tensors
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

            # apped additional inputs
            if additional_inputs is not None:
                system.update(additional_inputs[idx])

            # apply transforms
            for transform in self.transforms:
                system = transform(system)

            batch.append(system)

        # collate batch in a dict of tensors
        batch = _atoms_collate_fn(batch)

        # Move input batch to device
        batch = {p: batch[p].to(self.device) for p in batch}

        return batch

    @abstractmethod
    def get_prior(
        self, inputs: Dict[str, torch.Tensor], t: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        defines the prior p(x_t) for the reverse diffusion process.

        Args:
            inputs: input data with x_0 for each target property.
            t: the start time step of the reverse process.
        """
        pass

    @abstractmethod
    def __call__(
        self,
        inputs: Dict[str, torch.Tensor],
        t: Optional[torch.Tensor] = None,
        sample_prior: bool = True,
    ) -> Tuple[
        Dict[str, torch.Tensor], Optional[list], Optional[torch.Tensor], Dict[str, Any]
    ]:
        """
        defines the reverse diffusion process for sampling.

        Args:
            inputs: input data with x_0 to sample from prior
                    or starting x_t for each target property.
                    Note: You need to pass dummy inputs for the target properties
                    if sampling from complete noise to inder priors.
            t: the start time step of the reverse process.
            sample_prior: if True, sample from the prior p(x_t),
                          otherwise start from the properties passed in the input.
        """
        raise NotImplementedError


class DDPMSampler(Sampler):
    """
    Defines the reverse process based on the DDPM model by Ho et al.
    Subclasses the base class 'Sampler'.
    """

    def __init__(
        self,
        denoiser: Union[str, nn.Module],
        noise_schedule: NoiseSchedule,
        target_props: List[str],
        prop_noise_keys: Dict[str, str],
        time_key: str = "diff_step",
        **kwargs,
    ):
        """
        Args:
            denoiser: denoiser or path to denoiser to use for the reverse process.
            noise_schedule: the noise schedule of the reverse process.
            target_props: the target properties to denoise or sample.
            prop_noise_keys: the noise key for each target property in the model output.
            time_key: the key for the time prediction.
        """
        super().__init__(denoiser, noise_schedule, target_props, **kwargs)
        self.prop_noise_keys = prop_noise_keys
        self.time_key = time_key

    def get_prior(
        self, inputs: Dict[str, torch.Tensor], t: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        overwrites the ``get_prior`` method of the base class ``Sampler``.

        Args:
            inputs: input data with x_0 for each target property.
            t: the start time step of the reverse process.
        """
        outputs = {}

        # get priors for each property
        for prop in self.target_props:
            # check if property is in inputs to infer shape and dtype of noise
            if prop not in inputs:
                raise KeyError(
                    f"Property '{prop}' not found in inputs."
                    f"You can pass a dummy input under the property key."
                    f"This is needed, e.g. to infer the shape and dtype of the noise."
                )

            # sample noise
            noise = torch.randn_like(inputs[prop])

            # Invariance trick for atom positions.
            if self.invariant and prop == properties.R:
                noise = batch_center_systems(
                    noise, inputs[properties.idx_m], inputs[properties.n_atoms]
                )

            # prior for t < T
            if t is not None:
                if not isinstance(t, torch.Tensor):
                    raise ValueError("t must be a torch.Tensor when not None.")

                t = t.to(self.device)

                # get x_0 from inputs
                try:
                    x_0 = (
                        inputs[f"original_{prop}"]
                        if f"original_{prop}" in inputs
                        else inputs[prop]
                    )
                except KeyError:
                    raise KeyError(
                        f"Neither '{prop}' nor 'original_{prop}' are found in inputs."
                        f"Make sure to set the target properties, when 't' != None."
                    )

                # broadcast t to all atoms if moleculewise t is given
                if len(t) == len(inputs[properties.n_atoms]):
                    t = t[inputs[properties.idx_m]]

                # get prior using the forward diffusion process
                noise_params = self.noise_schedule(
                    t,
                    keys=["sqrt_alpha_bar", "sqrt_beta_bar"],
                )
                outputs[prop], _ = forward_marginal_gaussian(
                    x_0,
                    noise_params["sqrt_alpha_bar"].unsqueeze(-1),
                    noise_params["sqrt_beta_bar"].unsqueeze(-1),
                    noise,
                )

            # prior for t = T, i.e. Isotropic Gaussian N(0, I).
            else:
                outputs[prop] = noise

        return outputs

    def inference_step(
        self, inputs: Dict[str, torch.Tensor], iter: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        One inference step for the model to get the time steps and noise prediction.

        Args:
            inputs: input data for noise prediction.
            iter: the current iteration of the reverse process.
        """
        # append the current time step to the model input
        time_steps = torch.full_like(inputs[properties.idx_m], fill_value=iter)
        inputs[self.time_key] = time_steps.float() / self.noise_schedule.T

        model_out = self.denoiser(inputs)  # type: ignore

        # get the noise outputs for each target property.
        # for invariant properties, it must be implicitly centered by the model!
        noise = {
            prop: model_out[self.prop_noise_keys[prop]].detach()
            for prop in self.target_props
        }

        return time_steps, noise

    def reverse_step(
        self,
        x_t: Dict[str, torch.Tensor],
        noise: Dict[str, torch.Tensor],
        time_steps: torch.Tensor,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Defines one step/iteration of the reverse diffusion process as in Ho et al.

        Args:
            x_t: the current state x_t of the reverse process.
            noise: the noise prediction for the current state.
            time_steps: the time steps for the current state.
            inputs: the inputs to the denoiser.
        """
        # get noise schedule parameters
        noise_params = self.noise_schedule(
            time_steps.unsqueeze(-1),
            keys=["inv_sqrt_alpha", "beta", "inv_sqrt_beta_bar", "sqrt_sigma"],
        )
        inv_sqrt_alpha_t = noise_params["inv_sqrt_alpha"]
        beta_t = noise_params["beta"]
        inv_sqrt_beta_t_bar = noise_params["inv_sqrt_beta_bar"]
        sqrt_sigma_t = noise_params["sqrt_sigma"]

        # add noise with variance \sigma (stochasticity) only for t!=0
        # otherwise return the mean \mu as the final sample
        sqrt_sigma_t *= time_steps.unsqueeze(-1) != 0

        x_t_1 = {}

        # denoise each target property
        for prop in self.target_props:
            # broadcast if needed
            if len(x_t[prop]) == 1:
                x_t[prop] = x_t[prop].unsqueeze(-1)
            if len(noise[prop]) == 1:
                noise[prop] = noise[prop].unsqueeze(-1)

            # get the mean \mu of the reverse kernel q(x_t-1 | x_t)
            mu = inv_sqrt_alpha_t * (
                x_t[prop] - (beta_t * inv_sqrt_beta_t_bar) * noise[prop]
            )

            # sample noise
            eps = torch.randn_like(noise[prop])

            # Invariance trick for atom positions.
            if self.invariant and prop == properties.R:
                if inputs is None:
                    raise ValueError(
                        "Inputs must be provided when using the invariant trick."
                    )
                eps = batch_center_systems(
                    eps, inputs[properties.idx_m], inputs[properties.n_atoms]
                )

            # sample x_t-1 from q(x_t-1 | x_t) using the reparameterization trick
            x_t_1[prop] = mu + sqrt_sigma_t * eps

            # squeeze back if needed
            x_t_1[prop] = x_t_1[prop].squeeze(-1)
            x_t[prop] = x_t[prop].squeeze(-1)
            noise[prop] = noise[prop].squeeze(-1)

        return x_t_1

    def __call__(
        self,
        inputs: Dict[str, torch.Tensor],
        t: Optional[torch.Tensor] = None,
        sample_prior: bool = True,
    ) -> Tuple[
        Dict[str, torch.Tensor], Optional[list], Optional[torch.Tensor], Dict[str, Any]
    ]:
        """
        Defines the reverse diffusion process based on the DDPM model by Ho et al.
        overwrites the ``__call__`` method of the base class ``Sampler``.

        Args:
            inputs: input data with x_0 to sample from prior
                    or starting x_t for each target property.
                    Note: You need to pass dummy inputs for the target properties
                    if sampling from complete noise to inder priors.
            t: the start time step of the reverse process.
            sample_prior: if True, sample from the prior p(x_t),
                          otherwise start from the properties passed in the input.
        """
        if t is not None and (not isinstance(t, torch.Tensor) or t.shape != (1,)):
            raise ValueError(
                "t must be a torch.Tensor with shape (1,) when not None."
                "Sampling using different starting steps is not supported yet for DDPM."
            )

        batch = {prop: val.clone() for prop, val in inputs.items()}

        # sample from the prior p(x_t)
        if sample_prior:
            x_t = self.get_prior(batch, t)
            batch.update(x_t)
        else:
            x_t = {prop: batch[prop] for prop in self.target_props}

        # center the positions of the atoms before starting
        if self.invariant and properties.R in x_t:
            x_t[properties.R] = batch_center_systems(
                x_t[properties.R], batch[properties.idx_m], batch[properties.n_atoms]
            )

        # set all atoms as neighbors and compute neighbors only once before starting.
        if not self.recompute_neighbors:
            batch = compute_neighbors(batch, cutoff=50000.0, device=self.device)

        # initializations
        hist = []

        start = int(t.item()) if t is not None else self.noise_schedule.T - 1

        for i in tqdm(range(start, -1, -1)):
            # update the neighbors list after each iteration if not all atoms neighbors
            if self.recompute_neighbors:
                batch = compute_neighbors(batch, cutoff=self.cutoff, device=self.device)

            # get the time steps and noise predictions
            time_steps, noise = self.inference_step(batch, i)

            # save history
            if self.save_progress and (i % self.progress_stride == 0):
                hist.append({prop: val.cpu().clone() for prop, val in x_t.items()})

            # perform one reverse step
            x_t = self.reverse_step(x_t, noise, time_steps, batch)

            # update the batch for next iteration
            batch.update(x_t)

        # check validity of the molecules
        if self.check_validity:
            validity_res = check_validity(
                batch, *self.bonds_data.values(), progress_bar=False
            )
        else:
            validity_res = {}

        # move results to cpu
        if self.results_on_cpu:
            x_t = {prop: val.cpu() for prop, val in x_t.items()}

        return x_t, hist, None, validity_res


class MoreRedSampler(DDPMSampler):
    """
    MoreRed reverse process by Kahouli et al. for sampling or denoising
    from a time-aware diffusion model. Subclasses 'DDPMSampler'.
    """

    def __init__(
        self,
        denoiser: Union[str, nn.Module],
        noise_schedule: NoiseSchedule,
        target_props: List[str],
        prop_noise_keys: Dict[str, str],
        time_predictor: Optional[Union[str, nn.Module]] = None,
        max_steps: Optional[int] = None,
        time_key: str = "diff_step",
        pred_time_key: str = "diff_step_pred",
        predict_all: bool = True,
        average_time: bool = True,
        convergence_step: int = 0,
        **kwargs,
    ):
        """
        Args:
            denoiser: denoiser or path to denoiser to use for the reverse process.
            noise_schedule: the noise schedule of the reverse process.
            target_props: the target properties to denoise or sample.
            prop_noise_keys: the noise key for each target property in the model output.
            time_predictor: Seperate Predictor or path for the diffusion time step.
                            To be used for 'MoreRed-ITP' and 'MoreRed-AS'.
                            'MoreRed-JT' uses the time prediction coupled in denoiser.
            max_steps: The maximum number of steps to run the reverse process.
            time_key: The time key for the input to the noise model.
            pred_time_key: The time key for the output of the time predictor.
            predict_all: If True, predict the time steps through all the trajectory,
                         otherwise predict only the initial step.
            average_time: If True, average the atomwise time predictions to molecule.
            convergence_step: The step at which the reverse process converge.
        """
        super().__init__(
            denoiser, noise_schedule, target_props, prop_noise_keys, time_key, **kwargs
        )
        self.time_predictor = time_predictor
        self.pred_time_key = pred_time_key
        self.predict_all = predict_all
        self.average_time = average_time
        self.convergence_step = convergence_step

        self.max_steps = max_steps or self.noise_schedule.T

        if self.time_predictor is not None:
            if isinstance(self.time_predictor, str):
                self.time_predictor = torch.load(
                    self.time_predictor, device=self.device
                ).eval()
            else:
                self.time_predictor = self.time_predictor.to(self.device).eval()
        elif not self.predict_all:
            raise ValueError("time_predictor must be provided when 'predict_all=True'")

        # intern variables
        self._itp_time_steps = None

    def inference_step(
        self, inputs: Dict[str, torch.Tensor], iter: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        One inference step for the model to get the time steps and noise prediction.

        Args:
            inputs: input data for noise prediction.
            iter: the current iteration of the reverse process.
        """
        # append the predicted step to the model input if seperate representation
        if self.time_predictor is not None:
            # predict the time steps for all or only the initial step if MoreRed-ITP
            if self.predict_all or iter == 0:
                time_steps = self.time_predictor(inputs)[self.pred_time_key].detach()  # type: ignore
                # save the initial time steps for MoreRed-ITP
                if iter == 0:
                    self._itp_time_steps = time_steps

            # fixed decreasing time steps if MoreRed-ITP
            else:
                if self._itp_time_steps is None:
                    raise ValueError("Initial time steps not provided for MoreRed-ITP.")
                time_steps = self._itp_time_steps - (iter + 1) / self.noise_schedule.T
                time_steps = torch.clamp(time_steps, 0.0, 1.0)

            inputs[self.time_key] = time_steps  # type: ignore
            model_out = self.denoiser(inputs)  # type: ignore

        # get the time steps from the model output if joint representation
        else:
            model_out = self.denoiser(inputs)  # type: ignore
            time_steps = model_out[self.pred_time_key].detach()

        # average atomwise time predictions to moleculewise time steps
        if self.average_time and len(time_steps) == len(inputs[properties.idx_m]):
            time_steps = scatter_mean(
                time_steps, inputs[properties.idx_m], inputs[properties.n_atoms]
            )

        # covert time steps to integers indices
        time_steps = torch.round(time_steps * self.noise_schedule.T).long()

        # clip time steps to be between 0 and T-1
        time_steps = torch.clamp(time_steps, 0, self.noise_schedule.T - 1)

        # broadcast time steps to all atoms if moleculewise time steps
        if len(time_steps) != len(inputs[properties.idx_m]):
            time_steps = time_steps[inputs[properties.idx_m]]

        # get the noise outputs for each target property.
        # for invariant properties, it must be implicitly centered by the model!
        noise = {
            prop: model_out[self.prop_noise_keys[prop]].detach()
            for prop in self.target_props
        }

        return time_steps, noise

    def __call__(
        self,
        inputs: Dict[str, torch.Tensor],
        t: Optional[torch.Tensor] = None,
        sample_prior: bool = True,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Optional[List[Dict[str, torch.Tensor]]],
        Optional[torch.Tensor],
        Dict[str, List],
    ]:
        """
        Defines the sampling reverse process for MoreRed.

        Args:
            inputs: input data with x_0 to sample from prior
                    or starting x_t for each target property.
                    Note: You need to pass dummy inputs for the target properties
                    if sampling from complete noise to inder priors.
            t: the start time step of the reverse process.
            sample_prior: if True, sample from the prior p(x_t),
                          otherwise start from the properties passed in the input.
        """
        if t is not None and not isinstance(t, torch.Tensor):
            raise ValueError("t must be a torch.Tensor when not None.")

        # copy inputs to avoid inplace operations
        batch = {prop: val.clone() for prop, val in inputs.items()}

        # sample from the prior p(x_t)
        if sample_prior:
            x_t = self.get_prior(batch, t)
            batch.update(x_t)
        else:
            x_t = {prop: batch[prop] for prop in self.target_props}

        # center the positions of the atoms before starting
        if self.invariant and properties.R in x_t:
            x_t[properties.R] = batch_center_systems(
                x_t[properties.R], batch[properties.idx_m], batch[properties.n_atoms]
            )

        # set all atoms as neighbors and compute neighbors only once before starting.
        if not self.recompute_neighbors:
            batch = compute_neighbors(batch, cutoff=50000.0, device=self.device)

        # initializations
        hist = []
        num_steps = (
            torch.ones(
                len(batch[properties.n_atoms]), dtype=torch.long, device=self.device
            )
            * self.max_steps
        )
        done = torch.tensor(
            [False] * len(batch[properties.n_atoms]), device=self.device
        )
        validity_res = {}

        i = 0
        pbar = tqdm(total=self.max_steps)
        while i < self.max_steps:
            # update the neighbors list after each iteration if not all atoms neighbors
            if self.recompute_neighbors:
                batch = compute_neighbors(batch, cutoff=self.cutoff, device=self.device)

            # get the time steps and noise predictions
            time_steps, noise = self.inference_step(batch, i)

            # save history
            if self.save_progress and (
                i % self.progress_stride == 0 or i == (self.max_steps - 1)
            ):
                hist.append({prop: val.cpu().clone() for prop, val in x_t.items()})
                hist[-1]["time_steps"] = time_steps.cpu()

            # perform one reverse step
            x_t_1 = self.reverse_step(x_t, noise, time_steps, batch)

            # get the mask for the non-converged atoms
            mask_not_done = ~done[batch[properties.idx_m]]

            # update non-converged atoms for the next iteration
            for prop in self.target_props:
                x_t[prop][mask_not_done] = x_t_1[prop][mask_not_done]

            # update the batch for next iteration
            batch.update(x_t)

            # use the average time step for convergence check
            done = (
                torch.round(
                    scatter_mean(
                        time_steps, batch[properties.idx_m], batch[properties.n_atoms]
                    )
                )
                <= self.convergence_step
            )

            # check validity of the molecules
            if self.check_validity:
                validity_res = check_validity(
                    batch, *self.bonds_data.values(), progress_bar=False
                )
                valid = torch.tensor(
                    validity_res["stable_molecules"],
                    device=self.device,
                    dtype=torch.bool,
                )
                done = done | valid

            i += 1
            pbar.update(1)

            # save the number of steps
            num_steps[done & (num_steps == self.max_steps)] = i

            # check if all molecules converged
            if done.all():
                break

        pbar.close()

        # move results to cpu
        if self.results_on_cpu:
            num_steps = num_steps.cpu()
            x_t = {prop: val.cpu() for prop, val in x_t.items()}

        return x_t, hist, num_steps, validity_res
