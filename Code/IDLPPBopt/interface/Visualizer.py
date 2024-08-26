from typing import List, Union
import numpy as np
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
from rdkit import Chem
from rdkit.Chem	import rdDepictor
from rdkit.Chem.Draw import	rdMolDraw2D

from IDLPPBopt.utils import extract_privileged_substructures


class AtomAttentionVisualizer:

    # _default_cmap_colors = cmget_cmap("autumn_r", 256)(np.linspace(0, 1, 256))
    # _default_cmap_colors[:192,3] = np.logspace(-4, 0, 192)
    # DEFAULT_CMAP = ListedColormap(_default_cmap_colors)
    DEFAULT_CMAP = "YlOrRd"

    def __init__(self, figsize: List[int]=None, cmap=None, vmin=None, vmax=None, substructure_colors=None):
        self.figsize = figsize or (400, 300)
        self.cmap = cmap or self.DEFAULT_CMAP
        self.vmin = vmin
        self.vmax = vmax
        self.substructure_colors = (substructure_colors or [mpl_colors.to_rgba(c) for c in mpl_cm.Set2.colors])

    def draw_svg(self, 
                 smiles: str, 
                 atom_weights: np.ndarray, 
                 substructures: Union[str,List[int]]='privileged',
                 *,
                 cmap=None,
                 substructure_colors=None):
        '''Return visualization of mol as SMILES'''
        # Prepare default values
        if cmap is None:
            cmap = self.cmap
        if substructure_colors is None:
            substructure_colors = self.substructure_colors
        if substructures is None:
            substructures = []
        elif isinstance(substructures, str) and substructures == "privileged":
            substructures = extract_privileged_substructures(smiles, atom_weights)

        # Define atom highlight colors
        norm = mpl_colors.Normalize(vmin=(self.vmin or np.min(atom_weights)), vmax=(self.vmax or np.max(atom_weights)*1.3))
        cmap = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
        atom_colors = {i: cmap.to_rgba(wgt) for i,wgt in enumerate(atom_weights)}
        atom_indices = sorted(atom_colors.keys())

        # Prepare substructures, define highlight colors
        mol = Chem.MolFromSmiles(smiles)
        bond_indices, bond_colors = [], {}
        for bond in mol.GetBonds():
            bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            for i, subs in enumerate(substructures):
                if bi in subs and bj in subs:
                    bond_index = bond.GetIdx()
                    bond_indices.append(bond_index)
                    bond_colors[bond_index] = substructure_colors[i % len(substructure_colors)]
                    break
        
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(self.figsize[0], self.figsize[1])
        drawer.SetFontSize(.7)
        drawer.DrawMolecule(
            rdMolDraw2D.PrepareMolForDrawing(mol), 
            highlightAtoms=atom_indices, 
            highlightAtomColors=atom_colors,
            highlightBonds=bond_indices, 
            highlightBondColors=bond_colors)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg