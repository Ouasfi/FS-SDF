
import copy
import os
import torch
import numpy as np
import mcubes
import time
from src import loss, models, utils,  meta_modules as meta 
import trimesh
def get_mesh(data_tuple):
    logits, path_i = data_tuple
    pred_mesh = ref.mesh_from_logits(logits)
    gt_mesh = trimesh.load(path_i+ '/isosurf_scaled.off', process=False)
    print(path_i)
    return (pred_mesh, gt_mesh)
def get_context(encoder, batch, dataset):
    inputs = batch.get('inputs')#.cuda()
    p_context,  context_y = utils.get_levelset(batch, dataset)
    context_x = encoder(p_context, inputs)
    return context_x, context_y
@torch.no_grad()
def generate_params(batched_model, batch, dataset, test_time_optim_steps):
    context_x, context_y = get_context(batched_model.encoder, batch, dataset)
    start_time = time.time()
    params = batched_model.decoder.generate_params(context_x, context_y, intermediate=False, num_meta_steps=test_time_optim_steps)
    print(f"Adaptation in {time.time() - start_time} seconds")
    return params

class MetaGenerator:
    def __init__(self,
                exp_name,
                dataset,
                checkpoint,
                threshold = 0.5,
                device = torch.device("cuda"),
                fast_lr = 1e-4,
                out_features = 1,
                inner_steps  = 5,
                lr_type ='per_parameter',
                resolution = 16,
                batch_points = 1000000):       
        self.threshold = threshold
        self.device = device
        self.resolution = resolution
        self.checkpoint_path = '{}/checkpoints/'.format( exp_name)
        self.batch_points = batch_points
        self.min = -0.5
        self.max = 0.5
        self.inner_steps = inner_steps
        self.grid_points_split = self.split_grid_points()
        self.minT, self.maxT = -0.1, 0.1
        self.exp_name = exp_name
        self.exp =  f'{exp_name}/evaluation_{checkpoint}_@256/generation'

        batched_model = self.get_model(out_features, 
                        fast_lr ,
                        inner_steps ,
                        lr_type )

        self.load_checkpoint( batched_model, checkpoint)
        self.original_model = copy.deepcopy(self.model)
        self.model.to(self.device)
        self.model.eval()
        self.ds = dataset
    def reset(self):
        self.model = copy.deepcopy(self.original_model).to(self.device)
        
    def get_model (self, out_features, 
                    fast_lr = 1e-4,
                    inner_steps  = 5,
                    lr_type ='per_parameter',):
        feature_size = (1 +  16 + 32 + 64 + 128 + 128 ) * 7
        encoder = models.ShapeNetPoints_sdf_encoder(self.device)
        decoder = meta.ReLUFC_(in_features=feature_size, out_features=out_features,
                        num_hidden_layers=2, hidden_features=256)
        loss_fn = loss.sdf_L1_loss
        meta_decoder =  meta.MetaSDF(decoder,
            init_lr=fast_lr, 
            num_meta_steps = inner_steps,
            loss =loss_fn,
            lr_type=lr_type,
            )
        batched_model = models.ShapeNetPoints_sdf_maml(encoder, meta_decoder)#.cuda()
        return batched_model
    def split_grid_points(self):
        grid_points = utils.create_grid_points_from_bounds(self.min, self.max, self.resolution)
        grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()

        a = self.max + self.min
        b = self.max - self.min

        grid_coords = 2 * grid_points - a
        grid_coords = grid_coords / b

        grid_coords = torch.from_numpy(grid_coords).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        return torch.split(grid_coords, self.batch_points, dim=1)
    
    def load_checkpoint(self, batched_model, checkpoint):
        path = f'{self.exp_name}/checkpoints/checkpoint_epoch_{checkpoint}.tar'
        state_dict = batched_model.state_dict()
        state_dict.update(torch.load(path)['model_state_dict'])
        batched_model.load_state_dict(state_dict)
        self.model = batched_model
    def generate_mesh(self, data, test_time_optim_steps):

        self.reset()
        inputs = data['inputs'].to(self.device)
        batch_size = inputs.shape[0]
        params = self.generate_params(data,  test_time_optim_steps)
        logits_list = []
        for points in self.grid_points_split:
            with torch.no_grad():
                pred =  self.model.forward_with_params(points.repeat(batch_size,1,1),inputs,params)
                logits =pred[:,0, :].tanh()
            logits_list.append(logits.squeeze(0).detach().cpu())

        logits = torch.cat(logits_list, dim=1).numpy()

        return -logits
    def mesh_from_logits(self, logits):
        logits = np.reshape(logits, (self.resolution,) * 3)

        # padding to ba able to retrieve object close to bounding box bondary
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        vertices, triangles = mcubes.marching_cubes(
            logits, threshold)

        # remove translation due to padding
        vertices -= 1

        # rescale to original scale
        step = (self.max - self.min) / (self.resolution - 1)
        vertices = np.multiply(vertices, step)
        vertices += [self.min, self.min, self.min]

        return trimesh.Trimesh(vertices, triangles)
    def get_mesh(self, data_tuple):
        logits, path_i = data_tuple
        pred_mesh = self.mesh_from_logits(logits)
        gt_mesh = trimesh.load(path_i+ '/isosurf_scaled.off', process=False)
        print(path_i)
        return (pred_mesh, gt_mesh)
    def save_mesh(self,data_tupel):
        logits, path_i = data_tupel

        mesh = self.mesh_from_logits(logits)
        path_split = path_i.split('/')
        export_path = f'{self.exp}/{path_split[-2]}/{path_split[-1]}' 

        if not os.path.exists(export_path):
            os.makedirs(export_path)

        mesh.export(export_path + 'surface_reconstruction.off')
    def get_context(self, batch):
        inputs = batch.get('inputs').to(self.device)
        p_context,  context_y = utils.get_levelset(batch, self.ds)
        context_x = self.model.encoder(p_context.to(self.device), inputs)
        return context_x, context_y.to(self.device)
    @torch.no_grad()
    def generate_params(self, batch,test_time_optim_steps):
        context_x, context_y = self.get_context(batch)
        start_time = time.time()
        params = self.model.decoder.generate_params(context_x, context_y, intermediate=False, num_meta_steps=test_time_optim_steps)
        print(f"Adaptation in {time.time() - start_time} seconds")
        return params