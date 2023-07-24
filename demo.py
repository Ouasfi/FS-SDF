from generation import MetaGenerator
import importlib 
import argparse
import torch
import src.utils as utils
from tqdm import tqdm
voxelized_data = importlib.import_module("if-net.models.data.voxelized_data_shapenet")
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
import numpy as np
from src import utils
from scipy.spatial import cKDTree as KDTree

class DemoDataset(torch.utils.data.Dataset):
    def __init__(self, root,save_path ,shapes, n_points, sigma = None ):
        self.filenames = [root + shape.split('.')[0] for shape in shapes]
        self.savepaths = [save_path + shape.split('.')[0] for shape in shapes]
        self.root = root
        self.sigma = sigma
        self.n_points = n_points
        bb_min = -0.5
        bb_max = 0.5
        self.res = 128
        self.grid_points = utils.create_grid_points_from_bounds(bb_min, bb_max, self.res)
        self.kdtree = KDTree(self.grid_points)
        #self.p = [ np.array(trimesh.load( f'{i}.ply', process = False).vertices.astype(np.float32)) for i in self.filenames ]
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = self.savepaths[idx]
        pointcloud = np.load(f'{self.filenames[idx]}/pointcloud.npz')['points']
        idx = np.random.randint(pointcloud.shape[0], size= self.n_points )
        input_points= pointcloud[idx, :] + self.sigma* np.random.randn(self.n_points,3)
                            
        occupancies = np.zeros(len(self.grid_points), dtype=np.int8)

        _, idx = self.kdtree.query(input_points)
        occupancies[idx] = 1
        input = np.reshape(occupancies, (self.res,)*3)
        p = input_points.copy()
        p[:, 0], p[:, 2] = input_points[:, 2], input_points[:, 0]
        data ={"inputs" : np.array(input, dtype=np.float32),
               'pointcloud': np.array(2*p, dtype=np.float32),
                'path': path  ## 2*p
        }
        return data
    # def get_data_for_evaluation(self, idx):
    #     shapename = self.filenames[idx]
    #     data_shape = np.load(f'{shapename}/pointcloud.npz')
    #     data_space = np.load(f'{shapename}/points.npz')
    #     return data_shape, data_space

def main(args):

    split_file  = open(args.data_path + args.split_file, 'r').readlines()
    split_file = [shape.strip() for shape in split_file]
    print("Split File: " , split_file)
    testset = DemoDataset(args.data_path,args.save_path,  split_file, args.pc_samples, sigma = args.std_noise )
                                           

    gen = MetaGenerator(exp_name    = args.exp,
                        dataset     = testset,
                        checkpoint  = args.checkpoint,
                        device      = torch.device("cuda"),
                        inner_steps = args.inner_steps,
                        lr_type     ='per_parameter',
                        resolution  = 256,
                        batch_points = 38000)

    test_loader_iter =torch.utils.data.DataLoader(testset, batch_size=args.batch_size)


    print('heere')
    for b_i, batch in tqdm(enumerate(test_loader_iter)):
        print(batch.keys())
        batchlogits = gen.generate_mesh( batch, test_time_optim_steps = args.inner_steps)
        paths       = batch.get('path')
        print(paths)
        data_tuples = [(logits, path) for logits, path in  zip(batchlogits, paths) ]
        #p = Pool(mp.cpu_count()) if len(paths) > mp.cpu_count() else Pool(len(paths)) ; print(len(paths))
        meshs = map(gen.save_mesh, data_tuples,paths)
        list(meshs)

if __name__ == "__main__":
    parser = utils.get_parser(mode = 'eval')
    parser.add_argument('-split_file' , default='test_demo.lst', type=str)
    args   = parser.parse_args()
    main(args)
    
        