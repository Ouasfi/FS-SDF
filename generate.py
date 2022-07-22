from generation import MetaGenerator
import importlib 
import argparse
import torch
voxelized_data= importlib.import_module("if-net.models.data.voxelized_data_shapenet")
import multiprocessing as mp
from multiprocessing import Pool
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Run Model')
    parser.add_argument('-inner_steps' , default = 5, type=int)
    parser.add_argument('-checkpoint' , default = 1, type=int)
    parser.add_argument('-resume', dest='resume', action='store_true')
    parser.add_argument('-exp', help='experiment name', type=str)
    parser.add_argument('-n_workers' , default=5, type=int)

    parser.add_argument('-c','--category' , default=None, type=str)
    parser.add_argument('-pc_samples' , default=300, type=int)
    parser.add_argument('-res' , default=32, type=int)
    parser.add_argument('-batch_size' , default=4, type=int)
    parser.add_argument('-std_noise' , default=None, type=float)
    parser.add_argument('-save_path' , default='experiments', type=str)
    parser.add_argument('-noisy', dest='noisy', action='store_true')
    args = parser.parse_args()
    
    category = args.category #if len(args.category)>1 else None 
    split_file = 'if-net/shapenet/cat_splits.npz' if category is not None else 'if-net/shapenet/split.npz'
    print(split_file)
    batch_size = args.batch_size
    inner_loss = None
    testset = voxelized_data.VoxelizedDataset('test', 
                                            data_path = 'if-net/shapenet/data/',
                                            voxelized_pointcloud= True,
                                            pointcloud_samples = args.pc_samples,
                                            res=args.res,
                                            sample_distribution=[0.5,0.5 ],
                                            sample_sigmas=[0.1, 0.01],
                                            use_sdf = True,
                                            num_sample_points=500,
                                            batch_size=batch_size,
                                            num_workers=args.n_workers,
                                            matching_model = False,
                                            split_file= split_file,
                                            category = category,
                                            noisy = args.noisy,
                                            std_noise = args.std_noise)

    gen = MetaGenerator(exp_name = args.exp,
                        dataset = testset,
                        checkpoint = args.checkpoint,
                        device = torch.device("cuda"),
                        inner_steps  = args.inner_steps,
                        lr_type ='per_parameter',
                        resolution = 256,
                        batch_points = 38000)

    test_loader_iter = testset.get_loader(shuffle=False).__iter__()



    for b_i, batch in enumerate(test_loader_iter):
        
        batchlogits = gen.generate_mesh( batch, test_time_optim_steps = args.inner_steps)
        paths = batch.get('path')

        data_tuples = [(logits, path) for logits, path in  zip(batchlogits, paths) ]
        #p = Pool(mp.cpu_count()) if len(paths) > mp.cpu_count() else Pool(len(paths)) ; print(len(paths))
        meshs = map(gen.save_mesh, data_tuples)
        list(meshs)
        