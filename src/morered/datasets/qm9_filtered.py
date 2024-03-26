import logging
import os
import pickle
from typing import Optional

import numpy as np
from schnetpack import properties
from schnetpack.data import AtomsLoader, load_dataset
from schnetpack.datasets import QM9
from tqdm import tqdm

__all__ = ["QM9Filtered"]


class QM9Filtered(QM9):
    """
    QM9 dataset with a filter on the number of atoms.
    Only molecules of specific size are loaded.
    """

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        n_atoms_allowed: Optional[int] = None,
        shuffle_train: bool = True,
        indices_path: str = "n_atoms_indices.pkl",
        n_overfit_molecules: Optional[int] = None,
        permute_indices: bool = False,
        repeat_indices: int = 0,
        **kwargs
    ):
        """
        Args:
            datapath: path to directory containing QM9 database.
            batch_size: batch size.
            n_atoms_allowed: the exact number of atoms of each molecule.
            shuffle_train: whether to shuffle the training set.
            indices_path: path to pickle file containing indices of molecules
                          with specific number of atoms.
            n_overfit_molecules: number of molecules to overfit on.
            permute_indices: whether to permute the indices of molecules with
                             specific number of atoms.
            repeat_indices: whether to repeat the indices of molecules
        """
        super().__init__(datapath=datapath, batch_size=batch_size, **kwargs)

        self.n_atoms_allowed = n_atoms_allowed
        self.indices_path = indices_path
        self.shuffle_train = shuffle_train
        self.n_overfit_molecules = n_overfit_molecules
        self.permute_indices = permute_indices
        self.repeat_indices = repeat_indices

    def setup(self, stage: Optional[str] = None):
        """
        Overwrites the ``setup`` method to load molecules with given number of atoms.
        """
        if self.data_workdir is None:
            datapath = self.datapath
        else:
            datapath = self._copy_to_workdir()

        # (re)load datasets
        if self.dataset is None:
            self.dataset = load_dataset(
                datapath,
                self.format,
                property_units=self.property_units,
                distance_unit=self.distance_unit,
                load_properties=self.load_properties,
            )

            # use only molecules with specific number of atoms
            if self.n_atoms_allowed is not None and self.n_atoms_allowed > 0:
                # load indices
                if os.path.exists(self.indices_path):
                    with open(self.indices_path, "rb") as file:
                        indices = pickle.load(file)
                else:
                    indices = {}

                # get indices of molecules with specific number of atoms
                if self.n_atoms_allowed in indices.keys():
                    indices = indices[self.n_atoms_allowed]
                else:
                    tmp = []
                    for i in tqdm(range(len(self.dataset))):
                        if self.dataset[i][properties.n_atoms] == self.n_atoms_allowed:  # type: ignore
                            tmp.append(i)
                    indices[self.n_atoms_allowed] = tmp

                    # persist indices
                    with open(self.indices_path, "wb") as file:
                        pickle.dump(indices, file)
                    indices = indices[self.n_atoms_allowed]

            # get all indices (with any number of atoms)
            else:
                indices = list(range(len(self.dataset)))

            # overfit on a subset of molecules
            if self.n_overfit_molecules is not None and self.n_overfit_molecules > 0:
                # permute indices before overfitting
                if self.permute_indices:
                    indices = np.random.permutation(indices).tolist()

                indices = indices[: self.n_overfit_molecules] * (
                    int(len(indices) / self.n_overfit_molecules)
                    + (len(indices) % self.n_overfit_molecules)
                )

                if self.repeat_indices > 1:
                    indices = indices * self.repeat_indices

                logging.warning(
                    "Overfitting on {} molecules with indices {}".format(
                        self.n_overfit_molecules, indices[: self.n_overfit_molecules]
                    )
                )

            # subset dataset
            self.dataset = self.dataset.subset(indices)

            # load and generate partitions if needed
            if self.train_idx is None:
                self._load_partitions()

            # partition dataset
            self._train_dataset = self.dataset.subset(self.train_idx)  # type: ignore
            self._val_dataset = self.dataset.subset(self.val_idx)  # type: ignore
            self._test_dataset = self.dataset.subset(self.test_idx)  # type: ignore

        self._setup_transforms()

    def train_dataloader(self) -> AtomsLoader:
        """
        get training dataloader
        """
        if self._train_dataloader is None:
            self._train_dataloader = AtomsLoader(
                self.train_dataset,  # type: ignore
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle_train,
                pin_memory=self._pin_memory is not None and self._pin_memory,
            )
        return self._train_dataloader
