import logging
import os
import pickle
from typing import Dict, Optional

import numpy as np
import schnetpack.transform as trn
import torch
from ase import Atoms, build
from ase.data import chemical_symbols, covalent_radii
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from tqdm import tqdm

import morered as mrd
from morered.bonds import allowed_bonds_dict, bonds1, bonds2, bonds3


def batch_center_systems(
    systems: torch.Tensor, idx_m: torch.Tensor, n_atoms: torch.Tensor, dim: int = 0
):
    """
    center batch of systems moleculewise to have zero center of geometry

    Args:
        systems (torch.tensor): batch of systems (molecules)
        idx_m (torch.tensor): the system id for each atom in the batch
        n_atoms (torch.tensor): number of atoms in each system
        dim (int): dimension to scatter over
    """
    mean = scatter_mean(systems, idx_m, n_atoms, dim=dim)
    return systems - mean[idx_m]


def scatter_mean(
    systems: torch.Tensor, idx_m: torch.Tensor, n_atoms: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """
    compute the mean of a batch of systems moleculewise

    Args:
        systems (torch.Tensor): batch of systems (molecules)
        idx_m (torch.Tensor): the system id for each atom in the batch
        n_atoms (torch.Tensor): number of atoms in each system
        dim (int): dimension to scatter over
    """
    # compute the number of atoms per system if not given.
    if n_atoms is None:
        _, n_atoms = torch.unique_consecutive(idx_m, return_counts=True)

        if len(n_atoms) != len(torch.unique(idx_m)):  # type: ignore
            raise ValueError(
                "idx_m of the same system must be consecutive."
                " Alternatively, pass n_atoms per system as input."
            )

    shape = list(systems.shape)
    shape[dim] = len(idx_m.unique())
    tmp = torch.zeros(shape, dtype=systems.dtype, device=systems.device)
    sum = tmp.index_add_(dim, idx_m, systems)
    if len(sum.shape) == 1:
        mean = sum / n_atoms
    elif len(sum.shape) == 2:
        mean = sum / n_atoms.unsqueeze(-1)
    else:
        mean = sum / n_atoms.unsqueeze(-1).unsqueeze(-1)
    return mean


def compute_neighbors(
    old_batch,
    neighbor_list_trn: Optional[trn.Transform] = None,
    cutoff=5.0,
    fully_connected=False,
    additional_keys=[],
    device=None,
):
    """
    function to compute the neighbors for a batch of systems

    Args:
        old_batch: batch of systems to compute the neighbors for
        neighbor_list_trn: transform to compute the neighbors
        cutoff: cutoff radius for the neighbor list
        fully_connected: if True, all atoms are connected to each other.
                            Ignores the cutoff.
        additional_keys: additional keys to be included in the new batch
        device: Pytorch device
    """
    if device is None:
        device = old_batch[properties.R].device

    # get the float precision
    f_dtype = old_batch[properties.R].dtype

    # initialize the neighbor list transform
    if fully_connected:
        neighbors_calculator = mrd.transform.AllToAllNeighborList()
    else:
        neighbors_calculator = neighbor_list_trn or trn.MatScipyNeighborList(
            cutoff=cutoff
        )

    batch = []

    # compute the neighbors for each molecule in the batch
    for j, i in enumerate(old_batch[properties.idx_m].unique()):
        mask = old_batch[properties.idx_m] == i
        inp = {
            properties.idx: old_batch[properties.idx][[j]].detach().cpu(),
            properties.n_atoms: old_batch[properties.n_atoms][[j]].detach().cpu(),
            properties.Z: old_batch[properties.Z][mask].detach().cpu(),
            properties.R: old_batch[properties.R][mask].detach().cpu(),
            properties.cell: old_batch[properties.cell][[j]].detach().to(f_dtype).cpu(),
            properties.pbc: old_batch[properties.pbc].view(-1, 3)[j].detach().cpu(),
        }

        inp = neighbors_calculator(inp)

        batch.append(inp)

    # create the new batch
    batch = _atoms_collate_fn(batch)

    # add additional keys
    batch.update({k: old_batch[k] for k in additional_keys})

    batch = {p: batch[p].to(device) for p in batch}

    return batch


def rmsd(reference: Atoms, sample: Atoms, keep_original=True):
    """
    Computes the root mean squared deviation betweeen molecules in a batch

     Args:
         reference: The reference positions.
         sample: Sampled or denoised positions.
         keep_original: If True, don't overwrite the positions of the reference.
    """
    # keep the original positions or overwrite them with the rotated positions
    if keep_original:
        tmp = sample.copy()
    else:
        tmp = sample

    # compute the rotation and translation that minimizes the rmsd
    build.minimize_rotation_and_translation(reference, tmp)

    # compute the rmsd
    diff = ((reference.positions - tmp.positions) ** 2).sum(-1).mean() ** 0.5

    return diff


def batch_rmsd(references: torch.Tensor, samples: Dict[str, torch.Tensor]):
    """
    Computes the RMSD between a batch of reference and sampled/denoised positions.

    Args:
        references: The reference positions.
        samples: Sampled or denoised positions.
    """
    res = []

    # loop over molecules/systems
    for m in samples[properties.idx_m].unique():
        # get the indices of the current molecule
        mask = samples[properties.idx_m] == m

        # get the positions and atomic numbers
        R = samples[properties.R][mask].detach().cpu().numpy()
        R_0 = references[mask].detach().cpu().numpy()
        Z = samples[properties.Z][mask].detach().cpu().numpy()

        # create ase.Atoms objects
        ref_mol = Atoms(positions=R_0, numbers=Z)
        mol = Atoms(positions=R, numbers=Z)

        # compute the rmsd or set it to NaN if fails, e.g. for very different structures
        try:
            diff = rmsd(ref_mol, mol)
        except Exception:
            diff = torch.nan

        res.append(diff)

    return torch.tensor(res, device=samples[properties.R].device)


def check_connectivity(inputs, relax_coef=1.17):
    """
    Simple and fast connectivity check between atom pairs (not all molecule).

    Args:
        inputs: batch of molecules
        relax_coef: relaxation coefficient for the covalent radii of the bonds
    """
    results = torch.zeros_like(inputs[properties.idx_m])

    sum_w_H = 0
    sum_wo_H = 0

    # loop over molecules in the input batch
    for m in tqdm(inputs[properties.idx_m].unique()):
        # get the atomic numbers and distances for the current molecule
        mask = inputs[properties.idx_m] == m
        at_num = inputs[properties.Z][mask]
        dis = inputs[properties.R][mask]

        # compute the Euclidean distance matrix between all atoms pairs
        dists = torch.cdist(dis, dis)

        # query the covalent radii distances for atoms pairs
        covalent_dists = relax_coef * (
            covalent_radii[at_num][:, None] + covalent_radii[at_num][None, :]
        )

        # check if atom pairs are connected
        connected = ((dists < torch.from_numpy(covalent_dists)) & (0.0 < dists)).any(
            dim=1
        )

        # compute connectivity with and without Hydrogens
        sum_w_H += connected.all().long()
        sum_wo_H += (connected | (at_num == 1)).all().long()

        results[mask] = connected.long()

    num_mols = len(inputs[properties.idx_m].unique()) * 1.0

    # return the results and the connectivity with and without Hydrogens
    return results, sum_w_H / num_mols, sum_wo_H / num_mols


def squared_euclidean_distance(a, b):
    """
    Efficiently compute the squared Euclidean distance between two sets of points.

    Args:
        a: first set of points
        b: second set of points
    """
    distance = (
        (a**2).sum(axis=1)[:, None] - 2 * np.dot(a, b.T) + (b**2).sum(axis=1)[None]
    )

    return np.where(distance < 0, np.zeros(distance.shape), distance)


def check_validity(
    inputs,
    m_bonds_1,
    m_bonds_2,
    m_bonds_3,
    allowed_bonds,
    bonds_relaxation=None,
    progress_bar=True,
):
    """
    Fast check for the validity of molecules in a batch, including mol connectivity.

    Args:
        inputs: batch of molecules
        m_bonds_1: matrix of covalent radii for single bonds
        m_bonds_2: matrix of covalent radii for double bonds
        m_bonds_3: matrix of covalent radii for triple bonds
        allowed_bonds: number of allowed bonds per atom
        bonds_relaxation: relaxation coefficients for the covalent radii
        progress_bar: show tqsm progress bar
    """
    # set default covalent radii relaxation coefficients
    bonds_relaxation = bonds_relaxation or [0.1, 0.05, 0.03]

    bonds = []
    stable_atoms = []
    stable_molecules = []
    stable_atoms_wo_h = []
    stable_molecules_wo_h = []
    connected = []
    connected_wo_h = []

    # create idx_m if one system is given
    if properties.idx_m not in inputs:
        inputs[properties.idx_m] = torch.zeros(
            len(inputs[properties.Z]), dtype=torch.int32
        )
        progress_bar = False

    # loop over molecules in the batch
    for m in tqdm(inputs[properties.idx_m].unique(), disable=not progress_bar):
        # get the atomic numbers and positions for the current molecule
        mask = inputs[properties.idx_m] == m
        R = inputs[properties.R][mask]
        Z = inputs[properties.Z][mask]
        if torch.is_tensor(R):
            R = R.detach().cpu().numpy()
        if torch.is_tensor(Z):
            Z = Z.detach().cpu().numpy()

        # get covalent radii for the atoms in the current molecule
        ex_bonds_1 = m_bonds_1[Z[None], Z[:, None]]
        ex_bonds_2 = m_bonds_2[Z[None], Z[:, None]]
        ex_bonds_3 = m_bonds_3[Z[None], Z[:, None]]

        # compute distance matrix
        dist = squared_euclidean_distance(R, R) ** 0.5
        np.fill_diagonal(dist, np.inf)

        # get bond types per atom
        bonds_ = np.where(dist < ex_bonds_1 + bonds_relaxation[0], 1, 0)
        bonds_ = np.where(dist < ex_bonds_2 + bonds_relaxation[1], 2, bonds_)
        bonds_ = np.where(dist < ex_bonds_3 + bonds_relaxation[2], 3, bonds_)

        bonds.append(bonds_)

        # check if molecule is stable
        total_bonds = bonds_.sum(1)
        stable_at = allowed_bonds[Z] == total_bonds
        stable_atoms.append(stable_at)
        stable_molecules.append(stable_at.all())

        # check if molecule is stable without hydrogen
        stable_at_wo_h = stable_at.copy()
        stable_at_wo_h[Z == 1] = True
        stable_atoms_wo_h.append(stable_at_wo_h)
        stable_molecules_wo_h.append(stable_at_wo_h.all())

        # check if ALL the molecule is connected
        # using the exponent of the adjacency matrix trick
        bonds_t = (bonds[-1]) + np.eye(bonds[-1].shape[0])
        bonds_t = bonds_t > 0
        for i in range(bonds_t.shape[0]):
            bonds_t = bonds_t.dot(bonds_t)
        connected.append(bonds_t.all(1).any())

        # check if molecule is connected without hydrogen
        bonds_t[:, Z == 1] = True
        connected_wo_h.append(bonds_t.all(1).any())

    results = {
        "bonds": bonds,
        "stable_atoms": stable_atoms,
        "stable_molecules": stable_molecules,
        "connected": connected,
        "stable_atoms_wo_h": stable_atoms_wo_h,
        "stable_molecules_wo_h": stable_molecules_wo_h,
        "connected_wo_h": connected_wo_h,
    }

    return results


def generate_bonds_data(save_path: Optional[str] = None, overwrite: bool = False):
    """
    generate the bonds data as connectivity matrix between possible atoms.

    Args:
        save_path: path to save the data
        overwrite: overwrite existing data
    """
    save_path = save_path or "./bonds.pkl"

    if os.path.exists(save_path) and not overwrite:
        logging.info("Bonds data already exists, skipping generation and reloading...")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    atoms = np.array(chemical_symbols)
    indices = np.arange(len(atoms))
    m_bonds_1 = np.ones((len(atoms), len(atoms))) * -np.inf
    m_bonds_2 = m_bonds_1.copy()
    m_bonds_3 = m_bonds_1.copy()
    allowed_bonds = np.zeros((len(atoms)), dtype=np.int32)

    # define the bonds types and allowed bonds per atom
    for at in atoms:
        for at2 in atoms:
            if at in bonds1 and at2 in bonds1[at]:
                m_bonds_1[indices[atoms == at], indices[atoms == at2]] = (
                    bonds1[at][at2] / 100.0
                )
            if at in bonds2 and at2 in bonds2[at]:
                m_bonds_2[indices[atoms == at], indices[atoms == at2]] = (
                    bonds2[at][at2] / 100.0
                )
            if at in bonds3 and at2 in bonds3[at]:
                m_bonds_3[indices[atoms == at], indices[atoms == at2]] = (
                    bonds3[at][at2] / 100.0
                )
        if at in allowed_bonds_dict:
            allowed_bonds[indices[atoms == at]] = allowed_bonds_dict[at]

    data = {
        "bonds_1": m_bonds_1,
        "bonds_2": m_bonds_2,
        "bonds_3": m_bonds_3,
        "allowed_bonds": allowed_bonds,
    }

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    return data
