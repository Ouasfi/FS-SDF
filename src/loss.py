import torch
def sdf_L1_loss(pred, query_y, sigma = None, **kwargs):
    """
    L1 loss on sdf. 
    """
    assert pred.shape[1]<3, 'You may need to transpose dimensions'
    l1_loss = torch.abs(pred[:,0, :].tanh() - query_y).sum(-1).mean()
    return l1_loss