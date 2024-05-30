import torch

def contrastive_loss(z_i, z_j, temperature=0.1):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    
    sim_matrix = torch.exp(torch.mm(z, z.t()) / temperature)
    mask = torch.eye(batch_size * 2, device=z.device).bool()
    sim_matrix.masked_fill_(mask, 0)
    
    pos_sim = torch.exp(torch.sum(z_i * z_j, dim=1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    
    loss = pos_sim / sim_matrix.sum(dim=1)
    loss = -torch.log(loss).mean()
    return loss
