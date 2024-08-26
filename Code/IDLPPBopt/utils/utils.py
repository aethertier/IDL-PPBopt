from typing import List, Set

import numpy as np
from scipy import stats
from rdkit import Chem
from pybel import readstring

from ..sarpy.SARpytools import Grinder, Structure


def filter_subsets(sets: List[Set]):
    '''Filters out subsets, only retaining the largest superset'''
    # Sort sets by length in descending order
    sorted_sets = sorted(sets, key=len)
    # Filter out subsets of the larger sets
    supersets = []
    while sorted_sets:
        largest_set = sorted_sets.pop()
        sorted_sets = [s for s in sorted_sets if not s.issubset(largest_set)]
        supersets.append(largest_set)
    return supersets


def find_substructures(smiles: str):
    '''Recursively decompose a structure into fragments'''

    def _rec(structures):
        if not structures:
            return []
        subs = []
        for struct in structures:
            subs.extend(grinder.getFragments(struct))
        return subs + _rec(subs)
    
    grinder = Grinder(3, 18)
    cmpd = Structure(readstring('smi', smiles))
    substructures = _rec([cmpd])
    return [subs.smiles for subs in substructures]


def find_privileged_substructures(smiles: str, atom_weights: np.ndarray):
    '''Identify high-attention substructures'''
    # Identify substructures
    substructures = tuple()
    for subs in find_substructures(smiles):
        mol = Chem.MolFromSmiles(smiles)
        pattern = Chem.MolFromSmarts(subs)
        substructures += mol.GetSubstructMatches(pattern)
    # Identify privileged substructures
    privileged_substructures = []
    for idcs in map(list, substructures):
        mask = np.zeros(atom_weights.shape, dtype=bool)
        mask[idcs] = True
        if mask.all():
            continue
        mwu = stats.mannwhitneyu(atom_weights[mask], atom_weights[~mask], alternative='greater')
        if mwu.pvalue < .05:
            privileged_substructures.append(idcs)
    # Filter privileged substructures (filter out subsets)
    if privileged_substructures:
        privileged_substructures = list(map(list, filter_subsets(list(map(set, privileged_substructures)))))
    return privileged_substructures


def extract_substructure_by_indices(mol: Chem.rdchem.Mol, atom_indices):
    '''Extract substructures by atom indices'''
    # Create an Editable molecule from the original molecule
    emol = Chem.EditableMol(Chem.Mol())
    # Add the selected atoms to the new molecule
    for idx in atom_indices:
        emol.AddAtom(mol.GetAtomWithIdx(idx))
    # Add bonds between the selected atoms
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in atom_indices and end_idx in atom_indices:
            # Get the indices of the new substructure molecule
            new_begin_idx = atom_indices.index(begin_idx)
            new_end_idx = atom_indices.index(end_idx)
            emol.AddBond(new_begin_idx, new_end_idx, bond.GetBondType())
    # Get the final molecule
    submol = emol.GetMol()
    return submol
