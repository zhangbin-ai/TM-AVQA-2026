import torch
from torch import nn
import torch.nn.functional as F


def contrastive_loss(visual, audio, temperature=0.2):

    visual = F.normalize(visual, dim=1)
    audio = F.normalize(audio, dim=1)
    
    similarity_matrix = torch.mm(visual, audio.t())
    similarity_matrix /= temperature

    labels = torch.arange(visual.size(0)).to(visual.device)

    loss_v_to_a = F.cross_entropy(similarity_matrix, labels)
    loss_a_to_v = F.cross_entropy(similarity_matrix.t(), labels)
    
    return (loss_v_to_a + loss_a_to_v) / 2




def contrastive_loss_Rebuttal(visual, audio, temperature=0.2, rebuttal_threshold=0.8):

    visual = F.normalize(visual, dim=1)
    audio = F.normalize(audio, dim=1)

    sim = torch.mm(visual, audio.t())
    B = sim.size(0)
    pos_sim = sim.diag()
    threshold = rebuttal_threshold * pos_sim.unsqueeze(1)
    mask = sim <= threshold
    diag_idx = torch.arange(B, device=sim.device)

    mask[diag_idx, diag_idx] = True
    sim_masked = sim.masked_fill(~mask, -1e9)
    sim_masked = sim_masked / temperature
    labels = torch.arange(B).to(sim.device)

    loss_v2a = F.cross_entropy(sim_masked, labels)
    loss_a2v = F.cross_entropy(sim_masked.t(), labels)

    return (loss_v2a + loss_a2v) / 2