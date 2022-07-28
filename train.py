#!/usr/bin/env python
# coding: utf-8
from src import loss
from src.utils import get_exp
import src.meta_modules as meta
from src.models import ShapeNetPoints_sdf_encoder, ShapeNetPoints_sdf_maml
from training import BatchedMetaTrainer
import importlib
voxelized_data= importlib.import_module("if-net.models.data.voxelized_data_shapenet")
from functools import partial
import src.utils as utils
import torch
import argparse
import re



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Run Model')
    parser.add_argument('-epochs' , default = 100, type=int)
    parser.add_argument('-add_epochs' , default = 0, type=int)
    parser.add_argument('-inner_steps' , default = 5, type=int)
    parser.add_argument('-pretrained_model', default='ShapeNetPoints_sdf_sep_', type=str)
    parser.add_argument('-fast_lr' , default = 1e-4, type=float)
    parser.add_argument('-lr' , default = 1e-4, type=float)
    parser.add_argument('-checkpoint' , default = 73, type=int)
    parser.add_argument('-p_enc', dest='pretrained_encoder', action='store_true')
    parser.add_argument('-p_dec', dest='pretrained_decoder', action='store_true')
    parser.add_argument('-freeze', dest='freeze', action='store_true')
    parser.add_argument("--local_rank", type=int)

    parser.add_argument('-c','--category' , default=None, type=str)
    parser.add_argument('-pc_samples' , default=300, type=int)
    parser.add_argument('-res' , default=32, type=int)
    parser.add_argument('-n_workers' , default=8, type=int)
    parser.add_argument('-batch_size' , default=8, type=int)
    parser.add_argument('-std_noise' , default=0., type=float)
    parser.add_argument('-noisy', dest='noisy', action='store_true')
    
    args = parser.parse_args()
    logger = utils.DataLog()
    logger.log_exp_args(args)
    batch_size = args.batch_size
    logger.log_kv('batch_size',batch_size )
    category = args.category 
    split_file = 'if-net/shapenet/cat_splits.npz' if category is not None else 'if-net/shapenet/split.npz'
    print(split_file)
    train_dataset = voxelized_data.VoxelizedDataset('train',
                                            data_path = 'if-net/shapenet/data/',
                                            voxelized_pointcloud= True,
                                            pointcloud_samples= args.pc_samples, 
                                            res=args.res,
                                            sample_distribution=[0.5,0.5 ],
                                            sample_sigmas=[0.1, 0.01],
                                            use_sdf = True,
                                            num_sample_points=50000,
                                            batch_size=batch_size,
                                            num_workers=args.n_workers,
                                            matching_model = False,
                                            split_file= split_file,
                                            category = category,
                                            noisy = args.noisy,
                                            std_noise = args.std_noise
                                            )
    val_dataset = voxelized_data.VoxelizedDataset('val',
                                            data_path = 'if-net/shapenet/data/',
                                            voxelized_pointcloud= True,
                                            pointcloud_samples = args.pc_samples,
                                            res=args.res,
                                            sample_distribution=[0.5,0.5 ],
                                            sample_sigmas=[0.1, 0.01],
                                            use_sdf = True,
                                            num_sample_points=50000,
                                            batch_size=batch_size,
                                            num_workers=args.n_workers,
                                            matching_model = False,
                                            split_file= split_file,
                                            category = category,
                                            noisy = args.noisy,
                                            std_noise = args.std_noise
                                            )


    ## Model
    feature_size = (1 +  16 + 32 + 64 + 128 + 128 ) * 7
    encoder = ShapeNetPoints_sdf_encoder()
    exp  = get_exp(train_dataset, f'{args.pretrained_model}_batched_maml_{args.epochs}')#'iVoxels_dist-1.0_sigmas-0.1_v32_mShapeNetPoints'
    logger.log_kv('experience', exp)
   
    decoder = meta.ReLUFC_(in_features=feature_size, out_features=1,
                        num_hidden_layers=2, hidden_features=256)
    batched_model = ShapeNetPoints_sdf_maml(encoder, decoder).cuda()
        
    

    
    metatrainer = BatchedMetaTrainer(batched_model, 'cuda', 
                                    train_dataset, 
                                    val_dataset, 
                                    exp,
                                    fast_lr = args.fast_lr, 
                                    outer_lr = args.lr, 
                                    checkpoint = args.checkpoint,
                                    val_subset = None, 
                                    val_batches = 15,
                                    pretrained_encoder= args.pretrained_encoder,
                                    pretrained_decoder= args.pretrained_decoder,
                                    freeze_encoder = args.freeze
                                    )


    logger.log['inner_loss'] = [loss.sdf_L1_loss ]
    logger.log['outer_loss'] = [loss.sdf_L1_loss ]
    logger.save_log(save_path = metatrainer.exp_path)
    metatrainer.metatrain(iterations = args.epochs +args.add_epochs, 
                fas = args.inner_steps, 
                lr_type = 'per_parameter',
                pretrained_model = args.pretrained_model,
                train_subset = None,
                inner_loss = loss.sdf_L1_loss ,
                outer_loss = loss.sdf_L1_loss )    