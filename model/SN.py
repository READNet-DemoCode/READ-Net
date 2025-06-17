import torch
import torch.nn as nn
import torch.nn.functional as F
from base import *

class FastMutualInformationEstimator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super(FastMutualInformationEstimator, self).__init__()
        self.feature_dim = feature_dim

        self.projector_x = nn.Linear(feature_dim, hidden_dim)
        self.projector_y = nn.Linear(feature_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, y):
        batch_size = x.size(0)

        x_avg = x.sum(dim=1) / x.size(1)
        y_avg = y.sum(dim=1) / y.size(1)

        x_proj = self.norm(self.projector_x(x_avg))
        y_proj = self.norm(self.projector_y(y_avg))

        joint = torch.cat([x_proj, y_proj], dim=1)

        idx = torch.randperm(batch_size, device=x.device)
        y_shuffled = y_proj[idx]
        marginal = torch.cat([x_proj, y_shuffled], dim=1)

        joint_score = self.net(joint)
        marginal_score = self.net(marginal)

        mi_estimate = joint_score.mean() - torch.logsumexp(marginal_score, dim=0) + torch.log(torch.tensor(batch_size, dtype=torch.float, device=x.device))
        
        return mi_estimate


class FastHierarchicalFeatureSeparation(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super(FastHierarchicalFeatureSeparation, self).__init__()
        
        self.feature_dim = feature_dim

        self.separator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )

        self.self_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8)
        self.norm_self_attn = nn.LayerNorm(feature_dim)
        
        self.cross_modal_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8)
        self.norm_cross_attn = nn.LayerNorm(feature_dim)

        self.input_norm = nn.LayerNorm(feature_dim)

        self.mi_d_n = FastMutualInformationEstimator(feature_dim)
        self.mi_e_d = FastMutualInformationEstimator(feature_dim)
        self.mi_e_n = FastMutualInformationEstimator(feature_dim)
        self.mi_edep_d = FastMutualInformationEstimator(feature_dim)
        self.mi_edep_enondep = FastMutualInformationEstimator(feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 2)
        )
        self.cls=SequentialLightweightClassifier()

        self.lambda0 = 1.0
        self.lambda1 = 1.0
        self.lambda2 = 1.0
        self.lambda3 = 0.1
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 0.01
    

    def compute_entropy(self, features):

        features_avg = features.mean(dim=1)
        features_norm = F.softmax(features_avg, dim=1)
        epsilon = 1e-8
        entropy = -torch.sum(features_norm * torch.log(features_norm + epsilon)) / features_norm.size(0)
        return entropy
    
    def forward(self, F_d, F_e, F_n, labels=None):
        batch_size, seq_len, feature_dim = F_d.shape
        F_d = self.input_norm(F_d)
        F_e = self.input_norm(F_e)
        F_n = self.input_norm(F_n)

        mi_d_n = self.mi_d_n(F_d, F_n)
        mi_e_d = self.mi_e_d(F_e, F_d)
        mi_e_n = self.mi_e_n(F_e, F_n)
        entropy_e = self.compute_entropy(F_e)

        L_MI1 = -self.lambda0 * mi_d_n - self.lambda1 * mi_e_d - self.lambda2 * mi_e_n + self.lambda3 * entropy_e

        F_e_flat = F_e.reshape(-1, feature_dim)
        features_combined = self.separator(F_e_flat)
        features_combined = features_combined.view(batch_size, seq_len, feature_dim * 2)
        
        F_eDep = features_combined[..., :feature_dim]
        F_eNonDep = features_combined[..., feature_dim:]
        
        F_eDep_t = F_eDep.permute(1, 0, 2)  # [seq_len, batch_size, feature_dim]
        
        attn_mask = None
        F_eDep_attn, _ = self.self_attention(F_eDep_t, F_eDep_t, F_eDep_t, need_weights=False, attn_mask=attn_mask)
        
        F_eDep_attn = F_eDep_attn.permute(1, 0, 2)  # [batch_size, seq_len, feature_dim]
        F_eDep_enhanced = self.norm_self_attn(F_eDep_attn + F_eDep)
        
        F_d_t = F_d.permute(1, 0, 2)
        F_eDep_t = F_eDep_enhanced.permute(1, 0, 2)
        F_eDep_cross_attn, _ = self.cross_modal_attention(F_eDep_t, F_d_t, F_d_t, need_weights=False)
        F_eDep_cross_attn = F_eDep_cross_attn.permute(1, 0, 2)
        F_eDep_cross = self.norm_cross_attn(F_eDep_cross_attn + F_eDep_enhanced)
        
        mi_edep_d = self.mi_edep_d(F_eDep, F_d)
        mi_edep_enondep = self.mi_edep_enondep(F_eDep, F_eNonDep)
        
        L_MI2 = mi_edep_d - mi_edep_enondep
        
        L_cls = torch.tensor(0.0, device=F_d.device)
        L_total = -L_MI1 + self.alpha * L_MI2
        
        if labels is not None:
            # F_eDep_avg = F_eDep_cross.mean(dim=1)
            logits = self.cls(F_eDep)
            L_cls = F.cross_entropy(logits, labels)
            predictions = torch.argmax(logits, dim=1)
            correct_predictions = (predictions == labels)
            num_correct = correct_predictions.sum().item()
            total_samples = labels.size(0)
            accuracy = num_correct / total_samples
            print(accuracy)
            L_total = L_cls

        losses = {
            'L_MI1': -L_MI1.item(),
            'L_MI2': L_MI2.item(),
            'L_cls': L_cls.item() if isinstance(L_cls, torch.Tensor) else L_cls,
            'L_total': L_total.item()
        }
        
        return F_eDep_cross, F_eNonDep, losses


class HierarchicalFeatureSeparation(nn.Module):
    def __init__(self, feature_dim=256):
        super(HierarchicalFeatureSeparation, self).__init__()
        self.model = FastHierarchicalFeatureSeparation(feature_dim)
        
    def forward(self, F_d, F_e, F_n, labels=None):
        return self.model(F_d, F_e, F_n, labels)

