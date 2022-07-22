import numpy as np
import torch
def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list
def get_levelset(batch, dataset):
    dataset = dataset.dataset if isinstance(dataset, torch.utils.data.dataset.Subset) else dataset
    paths = batch.get('path')
    p = [torch.from_numpy(dataset.get_level_set(path)).float() for path in paths]
    p = torch.stack(p)
    return (p, torch.zeros(p.shape[0], p.shape[1]))