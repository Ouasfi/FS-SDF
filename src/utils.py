import numpy as np
import torch
import os
import numpy as np
import scipy
import torch
import pickle
import csv
import argparse
def create_grid_points_from_bounds(minimun, maximum, res):
    '''Create a 3D grid

    Args:
        minimun (int): minimun value in the grid 
        maximum (int): maximun value in the grid
        res (int): resolution

    Returns:
        np.array: 3D grid from meshgrid output (X,Y,Z )
    '''    
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list
def get_levelset(batch, dataset, device = 'cuda'):
    '''

    Args:
        batch (dict): Batch dict from the dataloader
        dataset (torch Dataset ): dataset

    Returns:
        tupe(torch.tensor): levelset points and sdf
    '''    
    dataset = dataset.dataset if isinstance(dataset, torch.utils.data.dataset.Subset) else dataset
    paths = batch.get('path')
    p = batch.get('pointcloud')
    if p is None:
        p = torch.stack([torch.from_numpy(dataset.get_level_set(path)).float() for path in paths] )
    return (p.to(device), torch.zeros(p.shape[0], p.shape[1], device = device))

def freeze(model):
    '''Freeze model params

    Args:
        model (nn.Module): Model to freeze

    Returns:
        nn.Module: the same model with frozen params(requires_grad = False)
    '''    
    for p in model.parameters():
        p.requires_grad = False
    return model
def get_exp(dataset, model):
    '''A function to get model name as defined in IFNet (Chibane et al.)repository

    Args:
        dataset (Dataset): Dataset class
        model (str): Model name

    Returns:
        [type]: [description]
    '''    
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

def get_parser(mode:str):
    parser = argparse.ArgumentParser(
                            description='Run Model')
    parser.add_argument('-n_workers' , default=8, type=int)
    parser.add_argument('-inner_steps' , default = 5, type=int)
    parser.add_argument('-checkpoint' , default = 73, type=int)
    parser.add_argument('-add_epochs' , default = 0, type=int)
    parser.add_argument('-c','--category' , default=None, type=str)
    parser.add_argument('-pc_samples' , default=300, type=int)
    parser.add_argument('-res' , default=32, type=int)
    parser.add_argument('-batch_size' , default=8, type=int)
    parser.add_argument('-std_noise' , default=0., type=float)
    parser.add_argument('-noisy', dest='noisy', action='store_true')
    parser.add_argument('-data_path', default='if-net/shapenet/', type=str)
    
    if mode =='train':
        parser.add_argument('-epochs' , default = 100, type=int)
        parser.add_argument('-pretrained_model', default='ShapeNetPoints_sdf_sep_', type=str)
        parser.add_argument('-fast_lr' , default = 1e-4, type=float)
        parser.add_argument('-lr' , default = 1e-4, type=float)
        parser.add_argument('-p_enc', dest='pretrained_encoder', action='store_true')
        parser.add_argument('-p_dec', dest='pretrained_decoder', action='store_true')
        parser.add_argument('-freeze', dest='freeze', action='store_true')
        parser.add_argument("--local_rank", type=int)
    elif mode =='eval':
        parser.add_argument('-resume', dest='resume', action='store_true')
        parser.add_argument('-exp', help='experiment name', type=str)
        parser.add_argument('-save_path' , default='experiments', type=str)
    return parser