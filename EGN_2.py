import torch
import torch.nn as nn
import torch.nn.functional as F

class ChildFeatureDynamicRegularization(nn.Module):
    def __init__(self, seq_len=186, feature_dim=256, num_segments=30, mu=0.1):

        super(ChildFeatureDynamicRegularization, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.K = num_segments
        self.mu = mu
        

        self.input_norm_eDep = nn.LayerNorm(feature_dim)
        self.input_norm_eNonDep = nn.LayerNorm(feature_dim)
        
        self.child_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )

        self.child_gcn_weight = nn.Parameter(torch.FloatTensor(128, 128))
        nn.init.xavier_uniform_(self.child_gcn_weight)

        self.gcn_norm = nn.LayerNorm(128)

        self.child_projection = nn.Linear(128, feature_dim)

        self.proj_norm = nn.LayerNorm(feature_dim)

        time_diffs = torch.abs(torch.arange(num_segments).unsqueeze(1) - torch.arange(num_segments).unsqueeze(0))
        self.register_buffer('time_attention', torch.exp(-0.1 * time_diffs.float()))

        self.register_buffer('norm_factor', torch.sqrt(torch.tensor(128.0)))
    
    def _process_child_features_batch(self, F_eDep, F_eNonDep):
        batch_size, seq_len, feature_dim = F_eDep.shape
        
        F_eDep = self.input_norm_eDep(F_eDep)
        F_eNonDep = self.input_norm_eNonDep(F_eNonDep)
        
        segment_len = seq_len // self.K
        segment_indices = [(i * segment_len, 
                          (i + 1) * segment_len if i < self.K - 1 else seq_len) 
                          for i in range(self.K)]
        
        F_eDep_segments = []
        F_eNonDep_segments = []
        
        for start_idx, end_idx in segment_indices:
            seg_mean_dep = F_eDep[:, start_idx:end_idx].mean(dim=1)
            seg_mean_nondep = F_eNonDep[:, start_idx:end_idx].mean(dim=1)
            
            F_eDep_segments.append(seg_mean_dep.unsqueeze(1))
            F_eNonDep_segments.append(seg_mean_nondep.unsqueeze(1))
        
        F_eDep_segments = torch.cat(F_eDep_segments, dim=1)
        F_eNonDep_segments = torch.cat(F_eNonDep_segments, dim=1)
        
        F_eDep_encoded = self.child_mlp(F_eDep_segments)
        F_eNonDep_encoded = self.child_mlp(F_eNonDep_segments)
        
        return F_eDep_encoded, F_eNonDep_encoded, segment_indices
    
    def _compute_child_similarity_matrix(self, F_eDep_encoded, F_eNonDep_encoded):

        cross_sim_matrix = torch.bmm(F_eDep_encoded, F_eNonDep_encoded.transpose(1, 2)) / self.norm_factor

        cross_sim_matrix = F.softmax(cross_sim_matrix, dim=2)
        
        return cross_sim_matrix
    
    def _child_graph_update(self, cross_sim_matrix, F_eDep_encoded, F_eNonDep_encoded):

        A = cross_sim_matrix + torch.eye(self.K, device=cross_sim_matrix.device).unsqueeze(0)

        D_inv_sqrt = torch.sum(A, dim=2).pow(-0.5)
        D_inv_sqrt_matrix = torch.diag_embed(D_inv_sqrt)

        A_norm = torch.bmm(torch.bmm(D_inv_sqrt_matrix, A), D_inv_sqrt_matrix)

        F_eDep_graph = torch.bmm(A_norm, F_eDep_encoded @ self.child_gcn_weight)
        F_eNonDep_graph = torch.bmm(A_norm, F_eNonDep_encoded @ self.child_gcn_weight)
        
        F_eDep_norm = self.gcn_norm(F_eDep_graph)
        F_eNonDep_norm = self.gcn_norm(F_eNonDep_graph)
        
        F_eDep_updated = F.relu(F_eDep_norm)
        F_eNonDep_updated = F.relu(F_eNonDep_norm)
        
        return F_eDep_updated, F_eNonDep_updated
    
    def _compute_flow_control_loss(self, F_eDep_updated, F_eNonDep_updated):
        batch_size, K, dim = F_eDep_updated.shape
        

        F_eDep_flat = F_eDep_updated.reshape(batch_size, -1)
        F_eNonDep_flat = F_eNonDep_updated.reshape(batch_size, -1)
        
        F_eDep_centered = F_eDep_flat - F_eDep_flat.mean(dim=1, keepdim=True)
        F_eNonDep_centered = F_eNonDep_flat - F_eNonDep_flat.mean(dim=1, keepdim=True)
        
        var_eDep = torch.sum(F_eDep_centered ** 2, dim=1) / (K * dim - 1)
        var_eNonDep = torch.sum(F_eNonDep_centered ** 2, dim=1) / (K * dim - 1)

        cov_eDep_self = torch.sum(F_eDep_centered * F_eDep_centered, dim=1) / (K * dim - 1)
        cov_eNonDep_self = torch.sum(F_eNonDep_centered * F_eNonDep_centered, dim=1) / (K * dim - 1)

        cov_eDep_eNonDep = torch.sum(F_eDep_centered * F_eNonDep_centered, dim=1) / (K * dim - 1)
        
        flow_loss = (- cov_eDep_self / torch.sqrt(var_eDep + 1e-8) +
                     cov_eNonDep_self / torch.sqrt(var_eNonDep + 1e-8) +
                     self.mu * cov_eDep_eNonDep / torch.sqrt(var_eDep * var_eNonDep + 1e-8))
        
        return flow_loss.mean()
    
    def _update_child_features(self, F_eDep, F_eNonDep, F_eDep_updated, F_eNonDep_updated, segment_indices):
        batch_size, _, _ = F_eDep.shape
        updated_eDep = F_eDep.clone()
        updated_eNonDep = F_eNonDep.clone()
        
        projected_eDep = self.proj_norm(self.child_projection(F_eDep_updated))
        projected_eNonDep = self.proj_norm(self.child_projection(F_eNonDep_updated))
        
        for k, (start_idx, end_idx) in enumerate(segment_indices):
            segment_len = end_idx - start_idx
            
            segment_update_dep = projected_eDep[:, k:k+1].expand(-1, segment_len, -1)
            segment_update_nondep = projected_eNonDep[:, k:k+1].expand(-1, segment_len, -1)
            
            alpha = 0.9
            updated_eDep[:, start_idx:end_idx] = alpha * updated_eDep[:, start_idx:end_idx] + (1 - alpha) * segment_update_dep
            updated_eNonDep[:, start_idx:end_idx] = alpha * updated_eNonDep[:, start_idx:end_idx] + (1 - alpha) * segment_update_nondep
        
        return updated_eDep, updated_eNonDep
    
    def forward(self, F_eDep, F_eNonDep):

        F_eDep_encoded, F_eNonDep_encoded, segment_indices = self._process_child_features_batch(F_eDep, F_eNonDep)
        
        cross_sim_matrix = self._compute_child_similarity_matrix(F_eDep_encoded, F_eNonDep_encoded)
        
        F_eDep_updated, F_eNonDep_updated = self._child_graph_update(cross_sim_matrix,
                                                                     F_eDep_encoded, 
                                                                     F_eNonDep_encoded)
        
        flow_loss = self._compute_flow_control_loss(F_eDep_updated, F_eNonDep_updated)
        
        updated_eDep, updated_eNonDep = self._update_child_features(F_eDep,
                                                                   F_eNonDep, 
                                                                   F_eDep_updated, 
                                                                   F_eNonDep_updated,
                                                                   segment_indices)
        
        return updated_eDep, updated_eNonDep, flow_loss


def test_child_module():
    batch_size = 2
    seq_len = 186
    feature_dim = 256
    
    F_eDep = torch.randn(batch_size, seq_len, feature_dim)
    F_eNonDep = torch.randn(batch_size, seq_len, feature_dim)
    
    cfdr = ChildFeatureDynamicRegularization(seq_len=seq_len, feature_dim=feature_dim)
    
    updated_eDep, updated_eNonDep, flow_loss = cfdr(F_eDep, F_eNonDep)
    
    print(f"{F_eDep.shape}")
    print(f"{F_eNonDep.shape}")
    print(f"{updated_eDep.shape}")
    print(f"{updated_eNonDep.shape}")
    print(f"{flow_loss.item()}")

if __name__ == "__main__":
    test_child_module()