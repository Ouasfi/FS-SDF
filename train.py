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
def main(args):
    
    ## Logging utils
    logger      = utils.DataLog()
    logger.log_exp_args(args)
    logger.log_kv('batch_size',args.batch_size )
    split_file  = '/cat_splits.npz' if args.category is not None else '/split.npz'
    split_file = args.data_path + split_file
    print("Split File: ", split_file)
    
    ## Shapenet Dataset class
    train_dataset = voxelized_data.VoxelizedDataset('train',
                                            data_path            = args.data_path + '/data/',
                                            voxelized_pointcloud = True,
                                            pointcloud_samples   = args.pc_samples, 
                                            res                  = args.res,
                                            sample_distribution  = [0.5,0.5 ],
                                            sample_sigmas        = [0.1, 0.01],
                                            use_sdf              = True,
                                            num_sample_points    = 50000,
                                            batch_size           = args.batch_size,
                                            num_workers          = args.n_workers,
                                            matching_model       = False,
                                            split_file           = split_file,
                                            category             = args.category,
                                            noisy                = args.noisy,
                                            std_noise            = args.std_noise
                                            )
    val_dataset = voxelized_data.VoxelizedDataset('val',
                                            data_path             = args.data_path + '/data/',
                                            voxelized_pointcloud  = True,
                                            pointcloud_samples    = args.pc_samples,
                                            res                   = args.res,
                                            sample_distribution   = [0.5,0.5 ],
                                            sample_sigmas         = [0.1, 0.01],
                                            use_sdf               = True,
                                            num_sample_points     = 50000,
                                            batch_size            = args.batch_size,
                                            num_workers           = args.n_workers,
                                            matching_model        = False,
                                            split_file            = split_file,
                                            category              = args.category,
                                            noisy                 = args.noisy,
                                            std_noise             = args.std_noise
                                            )


    ## IFNet Model
    feature_size = (1 +  16 + 32 + 64 + 128 + 128 ) * 7
    encoder = ShapeNetPoints_sdf_encoder()
    exp  = get_exp(train_dataset, f'{args.pretrained_model}_batched_maml_{args.epochs}')#'iVoxels_dist-1.0_sigmas-0.1_v32_mShapeNetPoints'
    logger.log_kv('experience', exp)
   
    decoder = meta.ReLUFC_(in_features=feature_size, out_features=1, num_hidden_layers=2, hidden_features=256)
    batched_model = ShapeNetPoints_sdf_maml(encoder, decoder).cuda()
        
    

    ## Training class for metalearning 
    metatrainer = BatchedMetaTrainer(model        = batched_model, 
                                    device        = 'cuda', 
                                    train_dataset = train_dataset, 
                                    val_dataset   = val_dataset, 
                                    exp_name      = exp,
                                    fast_lr       = args.fast_lr, 
                                    outer_lr      = args.lr, 
                                    checkpoint    = args.checkpoint,
                                    val_subset    = None, 
                                    val_batches   = 15,
                                    pretrained_encoder = args.pretrained_encoder,
                                    pretrained_decoder = args.pretrained_decoder,
                                    freeze_encoder     = args.freeze
                                    )


    logger.log['inner_loss']  = [loss.sdf_L1_loss ]
    logger.log['outer_loss']  = [loss.sdf_L1_loss ]
    logger.save_log(save_path = metatrainer.exp_path)
    metatrainer.metatrain(
                                    iterations       = args.epochs +args.add_epochs, 
                                    fas              = args.inner_steps, 
                                    lr_type          = 'per_parameter',
                                    pretrained_model = args.pretrained_model,
                                    train_subset     = None,
                                    inner_loss       = loss.sdf_L1_loss ,
                                    outer_loss       = loss.sdf_L1_loss )    


if __name__ == "__main__":
    parser = parser = utils.get_parser(mode = 'train')
    args = parser.parse_args()
    main(args)
   