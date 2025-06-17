import torch
import torch.nn as nn
import torch.nn.functional as F

class DualConsistencyRegularization(nn.Module):
    def __init__(self, seq_len=915, feature_dim=256, num_segments=30, gamma=0.1, lambda_reg=0.5):
        super(DualConsistencyRegularization, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.K = num_segments
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        
        self.input_norm_d = nn.LayerNorm(feature_dim)
        self.input_norm_n = nn.LayerNorm(feature_dim)
        self.input_norm_e = nn.LayerNorm(feature_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )
        
        self.gcn_weight = nn.Parameter(torch.FloatTensor(128, 128))
        nn.init.xavier_uniform_(self.gcn_weight)
        
        self.gcn_norm = nn.LayerNorm(128)
        
        self.segment_projection = nn.Linear(128, feature_dim)
        self.projection_norm = nn.LayerNorm(feature_dim)
        
        time_diffs = torch.abs(torch.arange(num_segments).unsqueeze(1) - torch.arange(num_segments).unsqueeze(0))
        self.register_buffer('time_attention', torch.exp(-gamma * time_diffs.float()))
        self.register_buffer('norm_factor', torch.sqrt(torch.tensor(128.0)))
        
    def _compute_similarity_matrix(self, encoded):
        """计算相似度矩阵，优化了矩阵计算"""

        sim_matrix = torch.bmm(encoded, encoded.transpose(1, 2)) / self.norm_factor

        sim_matrix = sim_matrix * self.time_attention.unsqueeze(0)

        sim_matrix = F.softmax(sim_matrix, dim=2)

        return sim_matrix
    def _compute_regularization_loss(self, F_d_updated, F_n_updated, F_e_updated):
        """计算正则化损失，优化了批处理操作"""

        F_d_self = torch.bmm(F_d_updated, F_d_updated.transpose(1, 2))
        F_n_self = torch.bmm(F_n_updated, F_n_updated.transpose(1, 2))
        F_e_self = torch.bmm(F_e_updated, F_e_updated.transpose(1, 2))
        
        F_dn_cross = torch.bmm(F_d_updated, F_n_updated.transpose(1, 2))
        F_de_cross = torch.bmm(F_d_updated, F_e_updated.transpose(1, 2))
        F_ne_cross = torch.bmm(F_n_updated, F_e_updated.transpose(1, 2))

        batch_losses = (torch.diagonal(F_d_self, dim1=1, dim2=2).sum(dim=1) +
                       torch.diagonal(F_n_self, dim1=1, dim2=2).sum(dim=1) +
                       torch.diagonal(F_e_self, dim1=1, dim2=2).sum(dim=1)) - \
                      self.lambda_reg * (torch.diagonal(F_dn_cross, dim1=1, dim2=2).sum(dim=1) +
                                       torch.diagonal(F_de_cross, dim1=1, dim2=2).sum(dim=1) +
                                       torch.diagonal(F_ne_cross, dim1=1, dim2=2).sum(dim=1))
        
        return batch_losses.mean()
    
    def _process_features_batch(self, features_batch, input_norm):
        batch_size, seq_len, feature_dim = features_batch.shape
        
        normalized_features = input_norm(features_batch)
        
        segment_len = seq_len // self.K
        segment_indices = [(i * segment_len, 
                           (i + 1) * segment_len if i < self.K - 1 else seq_len) 
                           for i in range(self.K)]
        
        segments = []
        for start_idx, end_idx in segment_indices:
            seg_mean = normalized_features[:, start_idx:end_idx].mean(dim=1)
            segments.append(seg_mean.unsqueeze(1))
        
        segments = torch.cat(segments, dim=1)
        
        encoded = self.mlp(segments)
        
        return encoded, segment_indices

    def _graph_convolution(self, sim_matrix, encoded):

        A = sim_matrix + torch.eye(self.K, device=sim_matrix.device).unsqueeze(0)
        
        D_inv_sqrt = torch.sum(A, dim=2).pow(-0.5)
        
        D_inv_sqrt_matrix = torch.diag_embed(D_inv_sqrt)
        
        A_norm = torch.bmm(torch.bmm(D_inv_sqrt_matrix, A), D_inv_sqrt_matrix)
        
        updated = torch.bmm(A_norm, encoded @ self.gcn_weight)
        updated = self.gcn_norm(updated)
        updated = F.relu(updated)
        
        return updated

    def _update_original_features(self, features, updated_features, segment_indices):
        batch_size, _, _ = features.shape
        updated = features.clone()
        
        projected_features = self.segment_projection(updated_features)
        projected_features = self.projection_norm(projected_features)
        
        for k, (start_idx, end_idx) in enumerate(segment_indices):
            segment_len = end_idx - start_idx
            
            segment_update = projected_features[:, k:k+1].expand(-1, segment_len, -1)
            
            alpha = 0.9
            updated[:, start_idx:end_idx] = alpha * updated[:, start_idx:end_idx] + (1 - alpha) * segment_update
        
        return updated

    def forward(self, features_DF_ALL, features_NF_ALL, features_EF_ALL):

        F_d_encoded, segment_indices = self._process_features_batch(features_DF_ALL, self.input_norm_d)
        F_n_encoded, _ = self._process_features_batch(features_NF_ALL, self.input_norm_n)
        F_e_encoded, _ = self._process_features_batch(features_EF_ALL, self.input_norm_e)

        sim_matrix_d = self._compute_similarity_matrix(F_d_encoded)
        sim_matrix_n = self._compute_similarity_matrix(F_n_encoded)
        sim_matrix_e = self._compute_similarity_matrix(F_e_encoded)

        F_d_updated = self._graph_convolution(sim_matrix_d, F_d_encoded)
        F_n_updated = self._graph_convolution(sim_matrix_n, F_n_encoded)
        F_e_updated = self._graph_convolution(sim_matrix_e, F_e_encoded)

        reg_loss = self._compute_regularization_loss(F_d_updated, F_n_updated, F_e_updated)

        updated_DF = self._update_original_features(features_DF_ALL, F_d_updated, segment_indices)
        updated_NF = self._update_original_features(features_NF_ALL, F_n_updated, segment_indices)
        updated_EF = self._update_original_features(features_EF_ALL, F_e_updated, segment_indices)
        
        return updated_DF, updated_NF, updated_EF, reg_loss


def test_module():
    batch_size = 2
    seq_len = 186
    feature_dim = 256
    
    features_DF_ALL = torch.randn(batch_size, seq_len, feature_dim)
    features_NF_ALL = torch.randn(batch_size, seq_len, feature_dim)
    features_EF_ALL = torch.randn(batch_size, seq_len, feature_dim)
    

    dcr = DualConsistencyRegularization(seq_len=seq_len, feature_dim=feature_dim)

    updated_DF, updated_NF, updated_EF, reg_loss = dcr(features_DF_ALL, features_NF_ALL, features_EF_ALL)


    print(f"{features_DF_ALL.shape}")
    print(f"{updated_DF.shape}")
    print(f"{updated_NF.shape}")
    print(f"{updated_EF.shape}")
    print(f"{reg_loss.item()}")

if __name__ == "__main__":
    test_module()