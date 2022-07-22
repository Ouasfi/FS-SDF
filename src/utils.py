import numpy as np
import torch
import os
import numpy as np
import scipy
import torch
import pickle
import csv

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

def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model
def get_exp(dataset, model):
    return 'i{}_dist-{}sigmas-{}v{}_m{}'.format(  'PC' + str(dataset.pointcloud_samples) if dataset.voxelized_pointcloud else 'Voxels',
                                    ''.join(str(e)+'_' for e in dataset.sample_distribution),
                                       ''.join(str(e) +'_'for e in dataset.sample_sigmas),
                                                                dataset.res,model)

class DataLog:

    def __init__(self):
        self.log = {}
        self.max_len = 0
        
    def log_exp_args(self, parsed_args):
        args = vars(parsed_args) # makes it a dictionary
        for k in args.keys():
            self.log_kv(k, args[k])

    def log_kv(self, key, value):
        # logs the (key, value) pair
        if key not in self.log:
            self.log[key] = []
        self.log[key].append(value)
        if len(self.log[key]) > self.max_len:
            self.max_len = self.max_len + 1

    def save_log(self, save_path=None):
        save_path = self.log['save_dir'][-1] if save_path is None else save_path
        pickle.dump(self.log, open(save_path+'/log.pickle', 'wb'))
        with open(save_path+'/log.csv', 'w') as csv_file:
            fieldnames = self.log.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in range(self.max_len):
                row_dict = {}
                for key in self.log.keys():
                    if row < len(self.log[key]):
                        row_dict[key] = self.log[key][row]
                writer.writerow(row_dict)

    def get_current_log(self):
        row_dict = {}
        for key in self.log.keys():
            row_dict[key] = self.log[key][-1]
        return row_dict

    def read_log(self, log_path):
        with open(log_path) as csv_file:
            reader = csv.DictReader(csv_file)
            listr = list(reader)
            keys = reader.fieldnames
            data = {}
            for key in keys:
                data[key] = []
            for row in listr:
                for key in keys:
                    try:
                        data[key].append(eval(row[key]))
                    except:
                        None
        self.log = data