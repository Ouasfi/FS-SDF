
import torch
from .loss import sdf_L1_loss
from .utils import get_levelset
   

def train_batch(encoder, decoder, i, batch, dataloader,  optimizer, loss_fn = sdf_L1_loss):
            #torch.cuda.empty_cache()
    p_query = batch.get('grid_coords').cuda()#.to(device)
    occ_query = batch.get('occupancies')
    inputs = batch.get('inputs').cuda()
    p_context,  occ_context = get_levelset(batch, dataloader.dataset)
    f_context = encoder(p_context.cuda(), inputs)
    f_query = encoder(p_query, inputs)
    meta_data = {'context': (f_context.unsqueeze(1),occ_context.cuda()), 'query': (f_query.unsqueeze(1), occ_query.cuda())}
    query_y = meta_data['query'][1]
    
    prediction, _ = decoder(meta_data, p_context = p_context)

    batch_loss = loss_fn(prediction, query_y,decoder.sigma_outer )
    
    print()
    print('Batch:', i , 'batch loss:',  batch_loss.item())
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss
def train_epoch(encoder, decoder, dataloader,  optimizer, loss_fn = sdf_L1_loss):
    encoder.train()
    decoder.train()
    epoch_train_loss = 0
    epoch_train_l1 = 0
    for i, batch in enumerate(dataloader):
        batch_loss= train_batch(encoder, decoder, i, batch, dataloader,  optimizer,loss_fn)
        epoch_train_loss += batch_loss.detach().item()

    epoch_train_loss/=len(dataloader)
    
    return epoch_train_loss
@torch.no_grad()
def val_epoch(encoder, decoder, val_dataset,loss_fn = sdf_L1_loss, num_batches = 15, subset = None):
    encoder.eval()
    decoder.eval()
    sum_val_loss = 0
    sum_val_l1 = 0
    for _ in range(num_batches):
        try:
            batch = val_data_iterator.next()
        except:
            val_data_iterator = val_dataset.get_loader(subset = subset, shuffle=False).__iter__()
            batch = val_data_iterator.next()

        p_query = batch.get('grid_coords').cuda()
        occ_query = batch.get('occupancies')
        inputs = batch.get('inputs').cuda()
        p_context,  occ_context = get_levelset(batch, val_dataset)
        f_context = encoder(p_context.cuda(), inputs)
        f_query = encoder(p_query.cuda(), inputs)
        meta_data = {'context': (f_context.unsqueeze(1),occ_context.cuda()), 'query': (f_query.unsqueeze(1), occ_query.cuda())}
        query_y = meta_data['query'][1]
        
        prediction, _ = decoder(meta_data, p_context = p_context)

        batch_loss = loss_fn(prediction, query_y, decoder.sigma_outer)

        print('val_batch loss:',  batch_loss.item())
        sum_val_loss += batch_loss.item()
        del p_query, occ_query , p_context,  occ_context, f_context, f_query, prediction, inputs
        del meta_data
        torch.cuda.empty_cache()
    sum_val_loss/=num_batches
    
    return sum_val_loss