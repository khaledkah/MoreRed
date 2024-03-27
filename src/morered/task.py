import inspect
import logging
from typing import Dict, Optional

import torch
from schnetpack.task import AtomisticTask, ModelOutput, UnsupervisedModelOutput
from torch import nn
from torchmetrics import Metric

log = logging.getLogger(__name__)


class DiffModelOutput(ModelOutput):
    """
    define diffusion output head.
    """

    def __init__(
        self,
        name: str,
        nll_metric: Optional[Metric] = None,
        **kwargs,
    ):
        """
        Args:
            name: name of the output.
            nll_metric: metric to compute the NLL.
        """
        super().__init__(name=name, **kwargs)

        if nll_metric is not None:
            self.nll_metrics = {
                "train": nll_metric,
                "val": nll_metric.clone(),
                "test": nll_metric.clone(),
            }
        else:
            self.nll_metrics = None

    def update_nll(
        self, inputs: Dict[str, torch.Tensor], subset: str
    ) -> Dict[str, torch.Tensor]:
        """
        update the NLL metric.

        Args:
            inputs: input batch.
            subset: the dataset split used.
        """
        if self.nll_metrics is not None:
            return self.nll_metrics[subset](inputs)
        else:
            return {}

    def calculate_loss(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        calculate the loss.

        Args:
            pred: outputs.
            target: target values.
        """
        if self.loss_weight == 0 or self.loss_fn is None:
            return torch.tensor(0.0)

        # extract the extra arguments of the loss function if needed
        args_ = inspect.getfullargspec(self.loss_fn).args[2:]
        kwargs = {k: pred[k] for k in args_ if k in pred}

        # calculate the loss using the extra arguments if needed
        if kwargs:
            loss = self.loss_weight * self.loss_fn(
                pred[self.name], target[self.target_property], **kwargs
            )
        else:
            loss = self.loss_weight * self.loss_fn(
                pred[self.name], target[self.target_property]
            )

        return loss


class DiffusionTask(AtomisticTask):
    """
    Defines the diffusion task for pytorch lightning.
    Subclasses the atomistic task and adds the diffusion NLL.
    """

    def __init__(
        self,
        diffuse_property: str,
        skip_exploding_batches: bool = True,
        include_l0: bool = False,
        time_key: str = "t",
        noise_key: str = "eps",
        noise_pred_key: str = "eps_pred",
        t0_noise_key: str = "eps_init",
        t0_noise_pred_key: str = "eps_init_pred",
        **kwargs,
    ):
        """
        Args:
            diffuse_property: property to diffuse.
            skip_exploding_batches: ignore exploding batches during training.
            include_l0: always add one extra forward pass to compute the noise at t=0.
                         Used to compute the l_0 term of the NLL., but slows training.
            time_key: key of the true diffusion time step in the input dictionary.
            noise_key: key of the true noise in the input dictionary.
            noise_pred_key: key of the predicted noise in the output dictionary.
            t0_noise_key: key of the true noise at t=0 in the input dictionary.
            t0_noise_pred_key: key of the predicted noise at t=0 in the output.
        """
        super().__init__(**kwargs)

        self.diffuse_property = diffuse_property
        self.skip_exploding_batches = skip_exploding_batches
        self.include_l0 = include_l0
        self.time_key = time_key
        self.noise_key = noise_key
        self.noise_pred_key = noise_pred_key
        self.t0_noise_key = t0_noise_key
        self.t0_noise_pred_key = t0_noise_pred_key

    def setup(self, stage=None):
        """
        overwrite the pytorch lightning task setup function.
        """
        # call the parent atomistic task setup
        AtomisticTask.setup(self, stage=stage)  # type: ignore

        # force some post-processing transforms during training
        forced_postprocessors = []
        for pp in self.model.postprocessors:
            if hasattr(pp, "force_apply"):
                if pp.force_apply:
                    forced_postprocessors.append(pp)
        self.model.forced_postprocessors = nn.ModuleList(forced_postprocessors)

    def predict_without_postprocessing(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        predict without post-processing transforms.
        Note: forced post-processing transforms will still be applied.

        Args:
            batch: input batch.
        """
        tmp_postprocessors = self.model.postprocessors
        self.model.postprocessors = self.model.forced_postprocessors
        pred = self(batch)
        self.model.postprocessors = tmp_postprocessors

        return pred

    def forward_t0(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extra forward pass to compute the noise at t=0.
        Usually to compute the l_0 term of the NLL but slows training.

        Args:
            batch: input batch.
        """
        # temporarly save the relavant quantities for L_t.
        tmp_diff_step = batch[self.time_key].clone()
        tmp_noise_pred = batch[self.noise_pred_key].clone()
        tmp_prop = batch[self.diffuse_property].clone()

        # load the initial noise and set the time step to 0.
        idx_t0 = torch.where((batch["ts"] == 0).all(0))[0][0]
        batch[self.diffuse_property] = batch[f"all_{self.diffuse_property}"][:, idx_t0]
        batch[self.time_key] = torch.zeros_like(batch[self.time_key])

        # feed-forward pass for t=0.
        # This will override the neighbors and distances computed for previous t.
        pred = self.predict_without_postprocessing(batch)

        # restore the initial (relevant) predictions for L_t for gradient backprop.
        batch[self.t0_noise_key] = batch[f"all_{self.noise_key}"][:, idx_t0]
        batch[self.t0_noise_pred_key] = pred[self.noise_pred_key]
        batch[self.time_key] = tmp_diff_step
        batch[self.noise_pred_key] = tmp_noise_pred
        batch[self.diffuse_property] = tmp_prop

        return batch

    def log_nll(self, inputs: Dict[str, torch.Tensor], subset: str):
        """
        log the diffusion  NLL.

        Args:
            inputs: input batch.
            subset: the dataset split used.
        """
        for output in self.outputs:
            if hasattr(output, "nll_metrics") and output.nll_metrics is not None:
                metrics = output.update_nll(inputs, subset)
                for key, val in metrics.items():
                    self.log(
                        f"{subset}_{output.name}_{key}",
                        val,
                        on_step=(subset == "train"),
                        on_epoch=(subset != "train"),
                        prog_bar=(subset != "train"),
                        metric_attribute=f"{subset}_{output.name}_{key}",
                    )

    def _step(self, batch: Dict[str, torch.Tensor], subset: str) -> torch.FloatTensor:
        """
        perform one forward pass and calculate the loss and log metrics.

        Args:
            batch: input batch.
            subset: the dataset split used.
        """
        # extract the target values from the batch
        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except Exception:
            pass

        # predict output quantity
        pred = self.predict_without_postprocessing(batch)

        # apply constraints
        pred, targets = self.apply_constraints(pred, targets)

        # calculate the loss
        loss = self.loss_fn(pred, targets)

        # log loss and metrics
        self.log(
            f"{subset}_loss",
            loss,
            on_step=(subset == "train"),
            on_epoch=(subset != "train"),
            prog_bar=(subset != "train"),
        )
        self.log_metrics(pred, targets, subset)

        # pefrom an extra forward pass for t=0 and log the NLL
        if self.include_l0:
            batch = self.forward_t0(batch)
        self.log_nll(batch, subset)

        return loss  # type: ignore

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[torch.FloatTensor]:
        """
        define the training step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        # perform forward pass
        loss = self._step(batch, "train")

        # skip exploding batches in backward pass
        if self.skip_exploding_batches and (
            torch.isnan(loss) or torch.isinf(loss) or loss > 1e10
        ):
            log.warning(
                f"Loss is {loss} for train batch_idx {batch_idx} and training step "
                f"{self.global_step}, training step will be skipped!"
            )
            return None

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.FloatTensor]:
        """
        define the validation step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        # enable non-training gradients with respect to specific quanitites if needed.
        torch.set_grad_enabled(self.grad_enabled)

        # forward pass
        loss = self._step(batch, "val")

        return {"val_loss": loss}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.FloatTensor]:
        """
        define the test step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        # enable non-training gradients with respect to specific quanitites if needed.
        torch.set_grad_enabled(self.grad_enabled)

        # forward pass
        loss = self._step(batch, "test")

        return {"test_loss": loss}
