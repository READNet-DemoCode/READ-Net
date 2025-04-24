import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class CrossModalAttention(nn.Module):

    def __init__(self, feature_dim, num_heads=8):
        super(CrossModalAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.o_proj = nn.Linear(feature_dim, feature_dim)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim)
        )

    def forward(self, x1, x2):

        batch_size, seq_len, _ = x1.shape

        q = self.q_proj(x1).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x2).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x2).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
        attn_output = self.o_proj(attn_output)

        x1 = self.norm1(x1 + self.dropout(attn_output))
        x1 = self.norm2(x1 + self.dropout(self.ffn(x1)))

        return x1


class AttentionBlock(nn.Module):
    def __init__(self, embed_size, heads, num_layers=2, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size, 
                nhead=heads, 
                dim_feedforward=embed_size*4,
                dropout=dropout,
                activation='gelu'
            ),
            num_layers=num_layers
        )
        
        self.conv_branch = nn.Sequential(
            nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_size),
            nn.GELU(),
            nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_size),
            nn.GELU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_size*2, embed_size),
            nn.LayerNorm(embed_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):

        x_trans = x.permute(1, 0, 2)
        x_trans = self.transformer(x_trans)
        x_trans = x_trans.permute(1, 0, 2)

        x_conv = x.permute(0, 2, 1)
        x_conv = self.conv_branch(x_conv)
        x_conv = x_conv.permute(0, 2, 1)

        x_combined = torch.cat([x_trans, x_conv], dim=2)
        x_fused = self.fusion(x_combined)
        
        return x_trans + x_fused
    
class LightweightClassifier(nn.Module):

    def __init__(self, feature_dim=256, hidden_dim=128):
        super(LightweightClassifier, self).__init__()

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(feature_dim)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

        self.single_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, X):

        if len(X.shape) == 3:

            attn_output, _ = self.temporal_attention(X, X, X)
            X = self.layer_norm(X + attn_output)

            X_t = X.transpose(1, 2)

            avg_pooled = self.global_avg_pool(X_t).squeeze(-1)
            max_pooled = self.global_max_pool(X_t).squeeze(-1)

            pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)

            logits = self.classifier(pooled_features)
            
        else:

            logits = self.single_classifier(X)
            
        return logits


class TemporalClassifier(nn.Module):

    def __init__(self, feature_dim, hidden_dim, num_classes, num_layers=1, dropout=0.3, bidirectional=True):
        super().__init__()

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.attention = TemporalAttention(
            hidden_dim * (2 if bidirectional else 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):

        gru_out, _ = self.gru(x)

        context, attention_weights = self.attention(gru_out)

        logits = self.classifier(context)
        
        return logits, attention_weights


class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):

        scores = self.projection(x).squeeze(-1)

        weights = F.softmax(scores, dim=1)

        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        
        return context, weights


class SequentialLightweightClassifier(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=128, num_classes=2):
        super().__init__()

        self.temporal_conv = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        self.bn = nn.BatchNorm1d(hidden_dim)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.temporal_conv(x)
        x = self.bn(x)
        x = F.gelu(x)

        avg_pooled = self.global_avg_pool(x).squeeze(-1)
        max_pooled = self.global_max_pool(x).squeeze(-1)

        x = torch.cat([avg_pooled, max_pooled], dim=1)

        logits = self.fc(x)
        
        return logits
    
class MultiscaleTemporalConvNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(MultiscaleTemporalConvNet, self).__init__()

        self.conv_scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, output_dim//3, kernel_size=k, stride=s, padding=k//2),
                nn.BatchNorm1d(output_dim//3),
                nn.GELU()
            ) for k, s in [(3,1), (5,2), (7,3)]
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv1d((output_dim // 3) * 3, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim),
            nn.GELU()
            )

        
    def forward(self, x):

        x = x.permute(0, 2, 1)

        multi_scale_features = []
        for conv in self.conv_scales:
            multi_scale_features.append(conv(x))

        adjusted_features = []
        target_size = multi_scale_features[0].size(-1)
        for feat in multi_scale_features:
            if feat.size(-1) != target_size:
                feat = F.interpolate(feat, size=target_size, mode='linear', align_corners=False)
            adjusted_features.append(feat)

        x = torch.cat(adjusted_features, dim=1)
        x = self.fusion(x)

        x = x.permute(0, 2, 1)
        return x

class ModalitySpecificEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, seq_len):
        super(ModalitySpecificEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        self.pos_encoder = nn.Parameter(
            torch.zeros(1, seq_len, hidden_dim)
        )
        nn.init.xavier_uniform_(self.pos_encoder)

        self.self_attn = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.LayerNorm(hidden_dim*4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*4, hidden_dim)
        )
        
    def forward(self, x):

        x = self.linear(x)
        x = self.norm(x)

        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]

        residual = x
        x, _ = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = self.norm1(x + residual)

        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        
        return x

class CrossModalAttentionImproved(nn.Module):
    def __init__(self, feature_dim):
        super(CrossModalAttentionImproved, self).__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*4),
            nn.LayerNorm(feature_dim*4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim*4, feature_dim)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, query_modal, key_modal):
        residual = query_modal

        q = self.query_proj(query_modal)
        k = self.key_proj(key_modal)
        v = self.value_proj(key_modal)

        scale = q.size(-1) ** 0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        context = torch.matmul(attn_weights, v)
        context = self.out_proj(context)
        context = self.dropout(context)

        out = self.norm1(residual + context)

        residual = out
        out = self.ffn(out)
        out = self.dropout(out)
        out = self.norm2(residual + out)
        
        return out

class HierarchicalTemporalAggregation(nn.Module):

    def __init__(self, feature_dim):
        super(HierarchicalTemporalAggregation, self).__init__()
        self.local_attn = nn.MultiheadAttention(feature_dim, 4, dropout=0.1)
        self.global_attn = nn.MultiheadAttention(feature_dim, 4, dropout=0.1)
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.gate = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape

        local_chunks = []
        window_size = min(32, seq_len)
        num_chunks = seq_len // window_size + (1 if seq_len % window_size != 0 else 0)
        
        for i in range(num_chunks):
            start_idx = i * window_size
            end_idx = min(start_idx + window_size, seq_len)
            chunk = x[:, start_idx:end_idx, :]

            chunk_out, _ = self.local_attn(chunk, chunk, chunk)
            chunk_out = self.norm1(chunk + chunk_out)
            local_chunks.append(chunk_out)
            
        local_features = torch.cat(local_chunks, dim=1)

        global_features, _ = self.global_attn(x, x, x)
        global_features = self.norm2(x + global_features)

        gate_input = torch.cat([local_features, global_features], dim=-1)
        gate = self.gate(gate_input)
        
        output = gate * local_features + (1 - gate) * global_features
        
        return output

