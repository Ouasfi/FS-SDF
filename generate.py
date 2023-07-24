from generation import MetaGenerator
import importlib 
import argparse
import torch
import src.utils as utils
from tqdm import tqdm
voxelized_data = importlib.import_module("if-net.models.data.voxelized_data_shapenet")
import multiprocessing as mp
from multiprocessing import Pool
def main(args):

    split_file  = '/cat_splits.npz' if args.category is not None else '/split.npz'
    split_file = args.data_path + split_file
    print("Split File: " , split_file)
    testset = voxelized_data.VoxelizedDataset('test', 
                                            data_path            = args.data_path + '/data/',
                                            voxelized_pointcloud = True,
                                            pointcloud_samples   = args.pc_samples,
                                            res                  = args.res,
                                            sample_distribution  = [0.5,0.5 ],
                                            sample_sigmas        = [0.1, 0.01],
                                            use_sdf              = True,
                                            num_sample_points    = 500,
                                            batch_size           = args.batch_size,
                                            num_workers          = args.n_workers,
                                            matching_model       = False,
                                            split_file           = split_file,
                                            category             = args.category,
                                            noisy                = args.noisy,
                                            std_noise            = args.std_noise)

    gen = MetaGenerator(exp_name    = args.exp,
                        dataset     = testset,
                        checkpoint  = args.checkpoint,
                        device      = torch.device("cuda"),
                        inner_steps = args.inner_steps,
                        lr_type     ='per_parameter',
                        resolution  = 256,
                        batch_points = 38000)

    test_loader_iter = testset.get_loader(shuffle=False).__iter__()


    print('heere')
    for b_i, batch in tqdm(enumerate(test_loader_iter)):
        
        batchlogits = gen.generate_mesh( batch, test_time_optim_steps = args.inner_steps)
        paths       = batch.get('path')

        data_tuples = [(logits, path) for logits, path in  zip(batchlogits, paths) ]
        #p = Pool(mp.cpu_count()) if len(paths) > mp.cpu_count() else Pool(len(paths)) ; print(len(paths))
        meshs = map(gen.save_mesh, data_tuples)
        list(meshs)

if __name__ == "__main__":
    parser = utils.get_parser(mode = 'eval')
    args   = parser.parse_args()
    main(args)
    
        