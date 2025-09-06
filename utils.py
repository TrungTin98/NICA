import numpy as np
import torch


def normalization (data, parameters=None):
    '''Normalize data in [0, 1] range.

    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
    '''
    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:

        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
            max_val[i] = np.nanmax(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)

        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                            'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            norm_data[:,i] = norm_data[:,i] - min_val[i]
            norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return norm_data, norm_parameters


def renormalization (norm_data, norm_parameters):
    '''Renormalize data from [0, 1] range to the original range.

    Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization

    Returns:
    - renorm_data: renormalized original data
    '''
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)
        renorm_data[:,i] = renorm_data[:,i] + min_val[i]

    return renorm_data


def rmse_loss (ori_data, imputed_data, data_m):
    '''Compute RMSE loss between ori_data and imputed_data

    Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness

    Returns:
    - rmse: Root Mean Squared Error
    '''

    ori_data, norm_parameters = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)

    # Only for missing values
    nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
    denominator = np.sum(1-data_m)

    rmse = np.sqrt(nominator/float(denominator))

    return rmse


def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.

    Args:
    - total: total number of samples
    - batch_size: batch size

    Returns:
    - batch_idx: batch index
    '''
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx


# create missing data and mask
def create_missing (data, rate=0.1):
    '''Create missing data and mask.'''
    n_rows, n_cols = data.shape
    num_points = n_rows * n_cols
    num_missing = int(num_points * rate)

    missing_data = data.copy()
    mask = np.ones(data.shape)

    while num_missing > 0:
        x = np.random.randint(low=0, high=n_rows)
        y = np.random.randint(low=0, high=n_cols)

        if missing_data[x, y] is not None:
            missing_data[x, y] = None
            mask[x, y] = 0
            num_missing -= 1

    return missing_data, mask


def random_corrupt(data, mask, rate=0.2):
    '''Randomly corrupt data.'''
    # Get the indices where mask is 1
    one_positions = np.argwhere(mask == 1)

    # Shuffle indices and pick 20% randomly
    np.random.shuffle(one_positions)
    num_to_drop = int(len(one_positions) * rate)
    drop_indices = one_positions[:num_to_drop]

    # Create copies of data and mask to modify
    corrupted_data = data.copy()
    corrupting_mask = mask.copy()

    # Drop the selected positions
    for i, j in drop_indices:
        corrupted_data[i, j] = 0
        corrupting_mask[i, j] = 0

    return corrupted_data, mask*(1-corrupting_mask)


# fill nan with mean
def nan_to_mean (missing_data):
    '''Fill nan with feature mean.'''
    n_cols = missing_data.shape[1]

    mean_vec = np.nanmean(missing_data, axis=0)

    filled_data = missing_data.copy()
    for i in range(n_cols):
        filled_data[:,i] = np.nan_to_num(filled_data[:,i], nan=mean_vec[i])

    return filled_data


# compute cosine similarity
def cosine (matrix1, matrix2, eps=1e-8):
  matrix1_norm = matrix1 / torch.max(matrix1.norm(dim=-1, keepdim=True), torch.tensor(eps))
  matrix2_norm = matrix2 / torch.max(matrix2.norm(dim=-1, keepdim=True), torch.tensor(eps))
  return torch.matmul(matrix1_norm, matrix2_norm.transpose(-2, -1))


# round data for categorical variables
def round_category (imputed_data, missing_data):
    '''Round categorical features.'''
    _, dim = missing_data.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = missing_data[~np.isnan(missing_data[:, i]), i]
        # Only for the categorical variable
        if (len(np.unique(temp))<20) and (np.sum(temp.astype(int)-temp)==0):
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data


# Loss function
def MSE_loss (data_1, data_2, mask):
    '''Calculate MSE loss between masked values in data1 and those in data2

    Args:
    - data_1: first data tensor
    - data_2: second data tensor
    - mask: mask tensor (nan ~ 0)

    Returns:
    - MSE loss
    '''
    return torch.mean((mask * data_1 - mask * data_2)**2) / torch.mean(mask * 1.)


def random_update_1d(input_tensor, rate=0.5):
    shape = input_tensor.shape

    if len(shape) == 2:
        rows, cols = shape
        row_values = torch.bernoulli(torch.full((rows, 1), 1-rate)).float()
        return row_values.repeat(1, cols)

    elif len(shape) == 3:
        matrices, rows, cols = shape
        row_values = torch.bernoulli(torch.full((matrices, rows, 1), 1-rate)).float()
        return row_values.repeat(1, 1, cols)

def random_update(input_tensor, rate=0.5):
    return torch.bernoulli(torch.full(input_tensor.shape, 1-rate)).float()