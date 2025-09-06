"""
This is the main script to run NICA on the datasets.
"""

import numpy as np
from copy import deepcopy
import argparse

from data_loader import data_loader
from nica_imputer import NICA_imputer
from utils import rmse_loss


def main (args):
    """Main function for imputation with NICA."""

    data_name = args.data_name
    miss_rate = args.miss_rate

    nica_config = {
        'n_versions': args.n_versions,
        'batch_size': args.batch_size,
        'grow_steps': args.grow_steps,
        'alpha_1'   : args.alpha_1,
        'alpha_2'   : args.alpha_2,
        'dropout'   : args.dropout,
        'n_iters'   : args.n_iters
    }

    data, missing_data, missing_mask = data_loader(data_name, miss_rate)

    nica = NICA_imputer(nica_config)
    nica.fit(deepcopy(missing_data))
    imputed_data = nica.impute(deepcopy(missing_data))

    rmse = rmse_loss(data, imputed_data, missing_mask)
    print('\nRMSE Performance: ' + str(np.round(rmse, 4)))
  
    return imputed_data, rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_name',
        choices=['airfoil', 'blood', 'boston', 'breast', 'california', 
                 'concrete', 'glass', 'iris', 'letter', 'spambase', 
                 'wine_red', 'wine_white', 'wine', 'yacht', 'yeast'],
        default='wine',
        type=str
    )
    parser.add_argument(
        '--miss_rate',
        help='missing proportion',
        default=0.4,
        type=float
    )

    parser.add_argument(
        '--n_versions',
        help='number of corrupted versions',
        default=8,
        type=int
    )
    parser.add_argument(
        '--batch_size',
        help='number of samples in each mini-batch',
        default=1024,
        type=int
    )
    parser.add_argument(
        '--grow_steps',
        help='number of growing steps',
        default=10,
        type=int
    )
    parser.add_argument(
        '--alpha_1',
        help='hyperparameter',
        default=10,
        type=float
    )
    parser.add_argument(
        '--alpha_2',
        help='hyperparameter',
        default=10,
        type=float
    )
    parser.add_argument(
        '--dropout',
        help='dropout rate',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--n_iters',
        help='number of training interations',
        default=1000,
        type=int
    )
    
    args = parser.parse_args() 
    
    # Calls main function  
    imputed_data, rmse = main(args)