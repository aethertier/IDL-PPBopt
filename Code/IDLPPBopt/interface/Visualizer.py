from typing import List
import numpy as np
import matplotlib
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize, ListedColormap
from rdkit import Chem
from rdkit.Chem	import rdDepictor
from rdkit.Chem.Draw import	rdMolDraw2D

from IDLPPBopt.utils import extract_privileged_substructures


class AtomAttentionVisualizer:

    # _default_cmap_colors = get_cmap("autumn_r", 256)(np.linspace(0, 1, 256))
    # _default_cmap_colors[:192,3] = np.logspace(-4, 0, 192)
    # DEFAULT_CMAP = ListedColormap(_default_cmap_colors)
    DEFAULT_CMAP = "YlOrRd"

    def __init__(self, smiles: str, atom_weights: np.ndarray, substructures: List[int]=None, *, 
                 cmap=None, vmin=None, vmax=None, bond_highlight_color=None, 
                 figsize: List[int]=None):
        self.smiles = smiles
        self.weights = atom_weights
        self.substructures = substructures or []
        self.cmap = cmap or self.DEFAULT_CMAP
        self.vmin = vmin
        self.vmax = vmax
        self.bond_highlight_color = bond_highlight_color or (.9, .5, 0., 1.)
        self.figsize = figsize or (400, 300)

    def highlight_privileged_substructures(self):
        self.substructures.extend(
            extract_privileged_substructures(self.smiles, self.weights))


    def draw_svg(self):
        '''Return visualization of mol as SMILES'''
        # Prepare molecule for 2d depiction
        mol = Chem.MolFromSmiles(self.smiles)
        rdDepictor.Compute2DCoords(mol)

        # Define atom highlight colors
        norm = Normalize(vmin=(self.vmin or self.weights.min()), 
                         vmax=(self.vmax or self.weights.max()))
        colo = ScalarMappable(norm=norm,cmap=self.cmap)
        
        atom_colors = {i: colo.to_rgba(wgt) for i,wgt in enumerate(self.weights)}
        atom_indices = sorted(atom_colors.keys())
        
        # Define bond highlight colors
        bond_indices, bond_colors = [], {}
        for bond in mol.GetBonds():
            i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            for subs in self.substructures:
                if i in subs and j in subs:
                    bndidx = bond.GetIdx()
                    bond_indices.append(bndidx)
                    bond_colors[bndidx] = self.bond_highlight_color
                    break

        drawer = rdMolDraw2D.MolDraw2DSVG(self.figsize[0], self.figsize[1])
        drawer.SetFontSize(.7)
        drawer.DrawMolecule(
            rdMolDraw2D.PrepareMolForDrawing(mol), 
            highlightAtoms=range(len(self.weights)),
            highlightAtomColors=atom_colors,
            highlightBonds=bond_indices,          
            highlightBondColors=bond_colors)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg