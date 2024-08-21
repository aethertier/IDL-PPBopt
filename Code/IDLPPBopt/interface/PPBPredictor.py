import os
import copy
from typing import List, Union
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
torch.manual_seed(8)
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.nn.Module.dump_patches = True

from AttentiveFP.AttentiveLayers import Fingerprint
from AttentiveFP.AttentiveLayers_viz import Fingerprint_viz
from AttentiveFP.getFeatures import get_smiles_dicts, get_smiles_array
from ..config import DEFAULT_MODEL_PATH


class PPBPredictor:

    MODEL_PATH = DEFAULT_MODEL_PATH
    MODEL_CLASS = Fingerprint

    def __init__(self, 
                 radius=2, 
                 T=2, 
                 fingerprint_dim=200, 
                 output_units_num=1, 
                 p_dropout=0.1):
        self.radius = radius
        self.T = T
        self.fingerprint_dim = fingerprint_dim
        self.output_units_num = output_units_num
        self.p_dropout = p_dropout
        self.smiles_series = None
        self.smiles_features = None
        self.model = None

    def load_smiles(self, smiles_series: pd.Series):
        ''' Load SMILES as a pandas series'''
        # Load SMILES
        smiles_series = pd.Series(smiles_series).apply(
            lambda smi: Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
        )
        n_smi = len(smiles_series)
        print(f"Loaded series of {n_smi:d} SMILES.")
        # Get SMILES features
        smiles_features = get_smiles_dicts(smiles_series) #, 'bla')
        smiles_series = smiles_series.loc[smiles_series.isin(smiles_features['smiles_to_atom_mask'].keys())]
        print(f"Calculated features for {len(smiles_series):d} of {n_smi:d} compounds.")
        # Save as class variables
        self.smiles_series = smiles_series
        self.smiles_features = smiles_features

    def prepare_model(self, model_path: Union[str, os.PathLike]=None):
        '''Prepare the predictive model'''
        assert self.smiles_series is not None \
            and self.smiles_features is not None
        # Get some input parameters    
        smile = self.smiles_series.iloc[:1]
        atm, bnd, atx, bdx, msk, smi2rdkit = get_smiles_array(smile, self.smiles_features)
        # Initialize model
        model = self.MODEL_CLASS(
                    radius = self.radius,
                    T = self.T,
                    input_feature_dim = atm.shape[-1], 
                    input_bond_dim = bnd.shape[-1],
                    fingerprint_dim = self.fingerprint_dim, 
                    output_units_num = self.output_units_num, 
                    p_dropout = self.p_dropout)
        model.cuda()
        # Transfer parameters from pretrained model
        best_model = torch.load(model_path or self.MODEL_PATH)
        best_model_wts = copy.deepcopy(best_model.state_dict())
        model.load_state_dict(best_model_wts)
        assert (best_model.align[0].weight == model.align[0].weight).all()
        self.model = model

    def evaluate(self, *, batch_size=64):
        '''Evaluate the SMILES given the model'''
        assert self.smiles_series is not None \
            and self.smiles_features is not None \
            and self.model is not None
        model = self.model
        model.eval()
        mol_predictions = []
        for i in range(0, len(self.smiles_series), batch_size):
            smiles_list = self.smiles_series.iloc[i:i+batch_size].values
            atm, bnd, atx, bdx, msk, smi2rdkit = get_smiles_array(smiles_list, self.smiles_features)
            atoms_prediction, mol_prediction = model(
                torch.Tensor(atm),
                torch.Tensor(bnd),
                torch.cuda.LongTensor(atx),
                torch.cuda.LongTensor(bdx),
                torch.Tensor(msk))
            mol_predictions.append(mol_prediction.data.squeeze().cpu().numpy())
        return np.concatenate(mol_predictions)
    

class PPBPredictorViz(PPBPredictor):

    MODEL_PATH = DEFAULT_MODEL_PATH
    MODEL_CLASS = Fingerprint_viz

    # def __init__(self, ...):              <-- inherited from parent

    # def load_smiles(self, ...):           <-- inherited from parent

    # def prepare_model(self, ...):         <-- inherited from parent

    def evaluate(self, *, batch_size=64):
        '''Evaluate the SMILES given the model'''
        assert self.smiles_series is not None \
            and self.smiles_features is not None \
            and self.model is not None
        model = self.model
        model.eval()
        mol_predictions, mol_attnweights = [], []
        for i in range(0, len(self.smiles_series), batch_size):
            smiles_list = self.smiles_series.iloc[i:i+batch_size].values
            # Original labels
            #   x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list
            atm, bnd, atx, bdx, msk, smi2rdkit = get_smiles_array(smiles_list, self.smiles_features)
            # Original labels
            # mol_prediction_array, atom_feature_array, atom_weight_array, mol_feature_array, mol_feature_unbounded_array
            _i, _ii, _iii, _iv, mol_weights, mol_pred = model(
                torch.Tensor(atm),
                torch.Tensor(bnd),
                torch.cuda.LongTensor(atx),
                torch.cuda.LongTensor(bdx),
                torch.Tensor(msk))
            mol_predictions.append(mol_pred.cpu().detach().squeeze().numpy())
            mol_weights = next(mol_weights[t].cpu().detach().numpy() for t in range(self.T))
            for smi, weigths in zip(smiles_list, mol_weights):
                idx = np.argsort(smi2rdkit[smi])
                mol_attnweights.append(weigths[idx].flatten())
        mol_predictions = np.concatenate(mol_predictions)
        return mol_predictions, mol_attnweights