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


def extract_privileged_substructures(smiles: str, atom_weights: np.ndarray):
    
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
