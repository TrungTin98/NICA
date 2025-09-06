"""
This file contains NICA imputer.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
import torch.nn as nn

from utils import random_corrupt, MSE_loss, round_category
from nica_model import NICA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class NICA_imputer(nn.Module):
    def __init__(self, config):
        super(NICA_imputer, self).__init__()

        # System Parameters
        self.n_versions = config['n_versions']
        self.batch_size = config['batch_size']
        self.grow_steps = config['grow_steps']
        self.alpha_1    = config['alpha_1']
        self.alpha_2    = config['alpha_2']
        self.dropout    = config['dropout']
        self.n_iters    = config['n_iters']

        # Losses
        self.losses     = []
        self.losses_obs = []
        self.losses_mis = []
        self.losses_rec = []


    def fit(self, data):
        mask = 1 - np.isnan(data)
        n_obsers, n_feats = data.shape

        # Normalization
        self.scaler = StandardScaler()
        transformed_data = self.scaler.fit_transform(data)
        fill_data = np.nan_to_num(transformed_data, 0)

        num_versions = self.n_versions
        cor_data_versions = []
        cor_mask_versions = []

        for _ in range(num_versions):
            fill_data_ = fill_data.copy()
            mask_      = mask.copy()

            corrupted_data, corrupting_mask = random_corrupt(fill_data_, mask_, rate=0.2)

            cor_data_versions.append(corrupted_data)
            cor_mask_versions.append(corrupting_mask)

        fill_data = torch.from_numpy(np.tile(fill_data, (num_versions, 1, 1))).to(device)
        cor_data  = torch.from_numpy(np.stack(cor_data_versions)).to(device)
        mask      = torch.from_numpy(np.tile(mask, (num_versions, 1, 1))).to(device)
        cor_mask  = torch.from_numpy(np.stack(cor_mask_versions)).to(device)


        # Model
        self.model = NICA(n_feats, grow_steps=self.grow_steps, dropout=self.dropout).to(device)

        nica_optim = torch.optim.Adam(self.model.parameters(), lr=0.001)

        #################### Training ####################
        for iter in tqdm(range(self.n_iters)):
            nica_optim.zero_grad()

            # Sample Batch
            total_idx = np.random.permutation(n_obsers)
            batch_idx = total_idx[:self.batch_size]

            X_0 = fill_data[:, batch_idx, :].clone()
            X_c = cor_data[:, batch_idx, :].clone()
            M   = mask[:, batch_idx, :].clone()
            M_c = cor_mask[:, batch_idx, :].clone()

            X_cK = self.model(X_c)

            # loss
            loss_rec = MSE_loss(X_cK, X_0, M_c)
            loss_obs = MSE_loss(X_cK, X_0, M)
            loss_mis = MSE_loss(X_cK, X_0, 1-M) * (-1)
            loss_mis = torch.clamp(loss_mis, min=-0.1)
            loss     = self.alpha_1*loss_rec + self.alpha_2*loss_obs + 1*loss_mis

            self.losses.append(loss.item())
            self.losses_rec.append(loss_rec.item())
            self.losses_obs.append(loss_obs.item())
            self.losses_mis.append(loss_mis.item())

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            nica_optim.step()
        #################### Finish training ####################


    def impute(self, data, num_impute=5):
        with torch.no_grad():
            self.model.eval()
            mask = torch.from_numpy(1 - np.isnan(data)).to(device)

            # Normalization
            transformed_data = self.scaler.transform(data)
            fill_data = torch.from_numpy(np.nan_to_num(transformed_data, 0)).to(device)

            output = torch.zeros_like(fill_data)
            for _ in range(num_impute):
                output = output + self.model(fill_data)
            output = output / num_impute

            imputed_data = mask * fill_data + (1-mask) * output

            # Renormalization
            imputed_data = self.scaler.inverse_transform(imputed_data.detach().cpu())

            # Rounding
            imputed_data = round_category(imputed_data, data)

            self.model.train()

        return imputed_data