"""
This file is for loading data and introducing missingness.
"""
# %%
import numpy as np
import pandas as pd
from utils import create_missing


def data_loader (data_name, miss_rate):
    """
    Load dataset and create missing data.
    """

    # Load data
    data_path = f'data/{data_name}.csv'
    data = pd.read_csv(data_path, header=None).values
    data = data.astype(np.float32)
    
    # Create missingness
    missing_data, missing_mask = create_missing(data, miss_rate)
    return data, missing_data, missing_mask

