import os, sys
from typing import Dict, List, Union
import pandas as pd
from . import PPBPredictor
from .interface.ArgumentParser import parse_arguments


def extract_smidata(df: pd.DataFrame, smi_column: str=None):
    '''Get SMILES data from parsed csv'''
    if smi_column is not None:
        # Select column with SMILES by column name
        smi_series = df[smi_column]
    elif df.shape[1] == 1:
        # If there is only one column...
        smi_series = df.iloc[:,0]
    else:
        # Select column with SMILES interactively
        colnames = list(df.columns)
        print('Columns:\n========')
        for i, col in enumerate(colnames):            
            print(f'{i:6d}  {col}')
        cix = int(input('\nSelect the column with SMILES:'))
        smi_series = df[colnames[cix]]
    return smi_series

def predict(smi_series):
    '''Run actual predictions of PPB bound fraction'''
    ppbpred = PPBPredictor()
    ppbpred.load_smiles(smi_series)
    ppbpred.prepare_model()
    y_pred = ppbpred.evaluate()
    return pd.Series(y_pred, index=smi_series.index)

def main():
    '''main function'''
    # Parse command line arguments
    args = parse_arguments(sys.argv[1:])
    # Read input csv
    df = pd.read_csv(args.infile)
    smi_series = extract_smidata(df, smi_column=args.smiles_column)
    # Run prediction
    y_pred = predict(smi_series)
    # Write output dataframe
    df[args.output_column] = y_pred
    # Output
    if args.outfile == '-':
        args.outfile = sys.stdout
    df.to_csv(args.outfile, float_format='%.4f', index=False)
    

if __name__ == '__main__':
    sys.exit(main())