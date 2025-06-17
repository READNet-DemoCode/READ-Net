import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(pred, target):

    return torch.norm(pred - target, p=2)


def adversarial_consistency_loss(E_d, E_e, E_n, margin_1=0.1, margin_2=0.05):

    cos_sim_de = torch.nn.functional.cosine_similarity(E_d, E_e, dim=-1)
    cos_sim_dn = torch.nn.functional.cosine_similarity(E_d, E_n, dim=-1)

    loss_de = torch.clamp(cos_sim_de - margin_1, min=0)
    loss_dn = torch.clamp(cos_sim_dn - margin_2, min=0)

    L_adv_cons = loss_de + loss_dn

    return L_adv_cons.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, features_A, features_B, features_C):

        features_A = F.normalize(features_A, p=2, dim=1)
        features_B = F.normalize(features_B, p=2, dim=1)
        features_C = F.normalize(features_C, p=2, dim=1)

        distance_AB = torch.norm(features_A - features_B, p=2, dim=1)
        distance_AC = torch.norm(features_A - features_C, p=2, dim=1)
        distance_BC = torch.norm(features_B - features_C, p=2, dim=1)


        loss_AB = F.softplus(self.margin - distance_AB)  
        loss_AC = F.softplus(self.margin - distance_AC)
        loss_BC = F.softplus(self.margin - distance_BC)

        total_loss = (loss_AB + loss_AC + loss_BC).mean()
        return total_loss
