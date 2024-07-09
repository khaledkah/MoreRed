import logging
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from schnetpack import properties
from torch import nn

from morered.sampling import Sampler
from morered.utils import batch_rmsd, check_validity, generate_bonds_data


class OutputWriterCallback(Callback):
    """
    Callback to store outputs of the model using ``torch.save`` for debugging.
    """

    def __init__(
        self,
        output_dir: str,
        only_exploding_outputs: bool = False,
        loss_threshold: float = 1e10,
    ):
        """
        Args:
            output_dir: output directory for prediction files.
            write_interval: one of ["batch", "epoch", "batch_and_epoch"].
            only_exploding_outputs: only write outputs that explode.
        """
        super().__init__()
        self.output_dir = output_dir
        self.only_exploding_outputs = only_exploding_outputs
        self.loss_threshold = loss_threshold
        os.makedirs(output_dir, exist_ok=True)

    def save_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        bdir: str,
    ):
        """
        Check for exploding loss and save (exploding) outputs.

        Args:
            outputs: model outputs.
            batch: input batch.
            batch_idx: batch index.
            bdir: save directory path.
        """
        key = "loss" if "loss" in outputs else "val_loss"
        exploded = False

        # check if loss is exploding
        if (
            key not in outputs
            or (outputs[key] > self.loss_threshold).any()
            or torch.isnan(outputs[key]).any()
            or torch.isinf(outputs[key]).any()
        ):
            exploded = True
            logging.warning(
                f"WARNING: exploded outputs in folder {bdir} "
                " with batch_idx {batch_idx}"
            )

        # save outputs
        if not self.only_exploding_outputs or exploded:
            os.makedirs(bdir, exist_ok=True)
            logs = {"inputs": batch, "outputs": outputs}
            torch.save(logs, os.path.join(bdir, f"{batch_idx}.pt"))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Save (exploding) outputs during training.
        Overwrites ``on_train_batch_end`` from ``Callback``.
        """
        bdir = os.path.join(self.output_dir, "train_" + str(trainer.current_epoch))
        self.save_outputs(outputs, batch, batch_idx, bdir)  # type: ignore

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """
        Save (exploding) outputs during validation.
        Overwrites ``on_validation_batch_end`` from ``Callback``.
        """
        bdir = os.path.join(self.output_dir, "val_" + str(trainer.current_epoch))
        self.save_outputs(outputs, batch, batch_idx, bdir)  # type: ignore


class SamplerCallback(Callback):
    """
    Callback to sample or denoise molecules for monitoring during training.
    """

    def __init__(
        self,
        sampler: Sampler,
        t: Optional[int] = None,
        max_steps: Optional[int] = None,
        sample_prior: bool = True,
        name: str = "sampling",
        store_path: str = "samples",
        every_n_batchs: int = 1,
        every_n_epochs: int = 1,
        start_epoch: int = 1,
        log_rmsd: bool = False,
        log_validity: bool = True,
        bonds_data_path: Optional[str] = None,
    ):
        """
        Args:
            sampler: sampler to be used for sampling/denoising.
            t: time step to start denoising. Defaults noise to start from prior.
            max_steps: maximum number of reverse steps when using MoreRed.
            sample_prior: whether to sample from the prior or use input as start sample.
            name: name of the callback.
            store_path: path to store the results and samples.
            every_n_batchs: sample every n batches.
            every_n_epochs: sample every n epochs.
            start_epoch: start sampling at this epoch.
            log_rmsd: whether to log the RMSD of denoised structures.
                      Useful for relaxation tasks of the atoms positions R.
            log_validity: whether to check and log the validity of the samples.
            bonds_data_path: path to the bonds data for validity checks.
        """
        super().__init__()
        self.sampler = sampler
        self.t = t
        self.max_steps = max_steps
        self.sample_prior = sample_prior
        self.name = name
        self.store_path = store_path
        self.every_n_batchs = every_n_batchs
        self.every_n_epochs = every_n_epochs
        self.start_epoch = start_epoch
        self.log_rmsd = log_rmsd
        self.log_validity = log_validity
        self.bonds_data = generate_bonds_data(bonds_data_path)

        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)

    def sample(
        self, model: nn.Module, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Samples or denoises a batch of molecules.

        Args:
            model: model to sample from.
            batch: input batch to be used for sampling/denoising.
        """
        # update the sampling model
        self.sampler.update_model(model)

        # sample from the prior
        if self.sample_prior:
            x_t = self.sampler.sample_prior(batch, self.t)
            batch.update(x_t)

        # sample / denoise
        samples, num_steps, hist = self.sampler(
            batch, t=self.t, max_steps=self.max_steps
        )

        # add important properties to save along with the sampled ones
        samples.update(
            {
                prop: val
                for prop, val in batch.items()
                if prop not in samples
                and prop
                in [
                    properties.R,
                    properties.Z,
                    properties.idx_m,
                    properties.idx,
                    properties.n_atoms,
                ]
            }
        )

        # move to cpu
        samples = {key: val.detach().cpu() for key, val in samples.items()}

        results = {
            "samples": samples,
            "hist": hist,
            "num_steps": (
                num_steps.cpu() if isinstance(num_steps, torch.Tensor) else num_steps
            ),
            "t": self.t.cpu() if isinstance(self.t, torch.Tensor) else self.t,
        }

        return results

    def save_samples(self, results: Dict[str, Any], epoch: int, test=False):
        """
        Saves the samples to disk.

        Args:
            results: dictionary containing the results and the samples to be saved.
            epoch: current epoch.
            test: whether the samples are from the test set.
        """
        phase = "test" if test else "val"

        # get the current file index
        self.file_idx = 0
        while os.path.exists(
            os.path.join(
                self.store_path,
                f"samples_{phase}_{epoch}_{self.file_idx}.pt",
            )
        ):
            self.file_idx += 1

        # save the samples
        with open(
            os.path.join(
                self.store_path,
                f"samples_{phase}_{epoch}_{self.file_idx}.pt",
            ),
            "wb",
        ) as f:
            torch.save(results, f)

    def _step(
        self,
        pl_module: LightningModule,
        batch: Dict[str, torch.Tensor],
        test: bool = False,
    ):
        """
        Perform a sampling step and log results.

        Args:
            model: model to sample from.
            batch: input batch to be used for sampling/denoising.
            test: whether the samples are from the test set.
        """
        # sample / denoise a batch
        results = self.sample(pl_module.model, batch)

        metrics = {}

        # check validity of the samples if requested
        if self.log_validity:
            validity_res = check_validity(results["samples"], *self.bonds_data.values())

            stable_ats = np.concatenate(validity_res["stable_atoms"])
            stable_mols = np.array(validity_res["stable_molecules"])
            stable_ats_wo_h = np.concatenate(validity_res["stable_atoms_wo_h"])
            stable_mols_wo_h = np.array(validity_res["stable_molecules_wo_h"])
            connected = np.array(validity_res["connected"])
            connected_wo_h = np.array(validity_res["connected_wo_h"])
            results["bonds"] = validity_res["bonds"]
            results["connectivity"] = torch.from_numpy(connected).cpu()
            results["stable_atoms"] = torch.from_numpy(stable_ats).cpu()
            results["stable_molecules"] = torch.from_numpy(stable_mols).cpu()
            results["stable_atoms_wo_h"] = torch.from_numpy(stable_ats_wo_h).cpu()
            results["stable_molecules_wo_h"] = torch.from_numpy(stable_mols_wo_h).to(
                "cpu"
            )
            results["connectivity_wo_h"] = torch.from_numpy(connected_wo_h).cpu()

            # infer metrics from validity results
            metrics = {
                "frac_stable_atoms": stable_ats.mean(),
                "frac_stable_molecules": stable_mols.mean(),
                "frac_stable_atoms_wo_h": stable_ats_wo_h.mean(),
                "frac_stable_molecules_wo_h": stable_mols_wo_h.mean(),
                "frac_connected_molecules": connected.mean(),
                "frac_connected_molecules_wo_h": connected_wo_h.mean(),
            }

            if "num_steps" in results and results["num_steps"] is not None:
                metrics["avg_num_sampling_steps"] = results["num_steps"].float().mean()
                metrics["med_num_sampling_steps"] = (
                    results["num_steps"].float().median()
                )
                metrics["std_num_sampling_steps"] = results["num_steps"].float().std()

                if hasattr(self.sampler, "max_steps") and self.sampler.max_steps is not None:  # type: ignore
                    metrics["frac_converged_sampling"] = (
                        (results["num_steps"] < self.sampler.max_steps).float().mean()  # type: ignore
                    )

        # compute RMSD of denoised structures if requested
        if self.log_rmsd:
            # get the reference positions
            reference_R = (
                batch[f"original_{properties.R}"]
                if "original_R" in batch
                else batch[properties.R]
            )

            res_rmsd = batch_rmsd(reference_R, results["samples"]).cpu()

            results["rmsd"] = res_rmsd
            metrics["rmsd"] = res_rmsd.mean()

        # log the metrics
        if metrics:
            for key, val in metrics.items():
                pl_module.log(
                    f"{'test' if test else 'val'}_{self.name}_{key}",
                    val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        # save the results and samples externally
        self.save_samples(results, pl_module.trainer.current_epoch, test=test)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """
        Overwrites ``on_validation_batch_end`` hook from ``Callback``.
        """
        # sample denoise only every n batches and m epochs
        if (
            trainer.current_epoch >= self.start_epoch
            and trainer.current_epoch % self.every_n_epochs == 0
            and batch_idx % self.every_n_batchs == 0
        ):
            self._step(pl_module, batch)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """
        Overwrites ``on_test_batch_end`` hook from ``Callback``.
        """
        self._step(pl_module, batch, test=True)
