
import torch
from .loss import sdf_L1_loss
from .utils import get_levelset
   

def train_batch(encoder, decoder, i, batch, dataloader,  optimizer, loss_fn = sdf_L1_loss):
    '''A function to perform a gradient step (inner and outer loop)

    Args:
        encoder (nn.Module): MetaModel encoder
        decoder (nn.Module): MetaModel decoder
        i (int): Iteration
        batch (dict): Batch of data from the dataloader
        dataloader (DataLoader): Dataloader
        optimizer (optim): Optimizer
        loss_fn (function, optional): Loss function. Defaults to sdf_L1_loss.

    Returns:
        torch.tensor   : Loss value of the current batch
    '''    
    p_query      = batch.get('grid_coords').cuda()#.to(device)
    occ_query    = batch.get('occupancies').cuda()
    inputs       = batch.get('inputs').cuda()
    p_context , occ_context = get_levelset(batch, dataloader.dataset, device = 'cuda')
    f_context   = encoder(p_context, inputs)
    f_query     = encoder(p_query, inputs)
    meta_data   = {'context': (f_context.unsqueeze(1) , occ_context), 
                   'query'  : (f_query.unsqueeze(1)   , occ_query)}
    query_y     = meta_data['query'][1]
    ## Inner steps are performed in the decoder forward pass
    prediction, _ = decoder(meta_data, p_context = p_context) # predictions on the query points after specialization
    batch_loss    = loss_fn(prediction, query_y,decoder.sigma_outer )
    
    print()
    print('Batch:', i , 'batch loss:',  batch_loss.item())
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss
def train_epoch(encoder, decoder, dataloader,  optimizer, loss_fn = sdf_L1_loss):
    '''Trains for one epoch

    Args:
        encoder (nn.Module): MetaModel encoder
        decoder (nn.Module): MetaModel decoder
        dataloader (DataLoader): Dataloader
        optimizer (optim): Optimizer
        loss_fn (function, optional): Loss function. Defaults to sdf_L1_loss.

    Returns:
        torch.tensor   :  Average Loss value over the dataloader
    '''    
    encoder.train()
    decoder.train()
    epoch_train_loss = 0
    for i, batch in enumerate(dataloader):
        batch_loss        = train_batch(encoder, decoder, i, batch, dataloader,  optimizer,loss_fn)
        epoch_train_loss += batch_loss.detach().item()

    epoch_train_loss/=len(dataloader)
    
    return epoch_train_loss
@torch.no_grad()
def val_epoch(encoder, decoder, val_dataset,loss_fn = sdf_L1_loss, num_batches = 15, subset = None):
    '''Validation over  one epoch

    Args:
        encoder (nn.Module): MetaModel encoder
        decoder (nn.Module): MetaModel decoder
        val_dataset (Dataset): Validation dataset
        loss_fn (function, optional): Loss function. Defaults to sdf_L1_loss.

    Returns:
        torch.tensor   :  Average Loss value over the dataloader
    '''    
    encoder.eval()
    decoder.eval()
    sum_val_loss = 0
    for _ in range(num_batches):
        try:
            batch = val_data_iterator.next()
        except:
            val_data_iterator = val_dataset.get_loader(subset = subset, shuffle=False).__iter__()
            batch             = val_data_iterator.next()

        p_query    = batch.get('grid_coords').cuda()
        occ_query  = batch.get('occupancies').cuda()
        inputs     = batch.get('inputs').cuda()
        p_context , occ_context = get_levelset(batch, val_dataset)
        f_context  = encoder(p_context, inputs)
        f_query    = encoder(p_query.cuda(), inputs)
        meta_data  = {'context': (f_context.unsqueeze(1) , occ_context), 
                      'query'  : (f_query.unsqueeze(1)   , occ_query)}
        query_y    = meta_data['query'][1]
        
        prediction, _ = decoder(meta_data, p_context = p_context)
        batch_loss    = loss_fn(prediction, query_y, decoder.sigma_outer)

        print('val_batch loss:',  batch_loss.item())
        sum_val_loss += batch_loss.item()
        del p_query, occ_query , p_context,  occ_context, f_context, f_query, prediction, inputs
        del meta_data
        torch.cuda.empty_cache()
    sum_val_loss/=num_batches
    
    return sum_val_loss