import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import *
from base import *
from EGN import *
from EGN_2 import *
from TS import *
from SN import *

class BranchNetworkNFN(nn.Module): 
    def __init__(self, input_dim):
        super(BranchNetworkNFN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=186, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(186)
        self.linear1 = nn.Linear(171, 128)
        self.ln1 = nn.LayerNorm(128)

        self.cross_attn1 = CrossModalAttention(feature_dim=128)
        self.cross_attn2 = CrossModalAttention(feature_dim=128)

        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.attn_block = AttentionBlock(embed_size=256, heads=8, num_layers=1, dropout=0.2)

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(256, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.Sigmoid()
        )
        
        
    def forward(self, X1, X2):
        X1 = self.conv1(X1)
        X1 = self.bn1(X1)
        X1 = F.gelu(X1)
        X1 = self.linear1(X1)
        X1 = self.ln1(X1)

        X1_enhanced = self.cross_attn1(X1, X2)
        X2_enhanced = self.cross_attn2(X2, X1)

        X = torch.cat([X1_enhanced, X2_enhanced], dim=2)
        X = self.fusion(X)
        X = self.attn_block(X)

        X_trans = X.permute(0, 2, 1)
        context = self.global_context(X_trans)
        X_trans = X_trans * context
        X_trans = X_trans.permute(0, 2, 1)

        X = X_trans[:, -1, :]
        return X_trans, X
    
class BranchNetworkDDFB(nn.Module):
    def __init__(self, output_dim):
        super(BranchNetworkDDFB, self).__init__()     

        self.video_feature_extractor = MultiscaleTemporalConvNet(input_dim=171, output_dim=128)

        self.video_encoder = ModalitySpecificEncoder(input_dim=128, hidden_dim=128, seq_len=915)  
        self.audio_encoder = ModalitySpecificEncoder(input_dim=128, hidden_dim=128, seq_len=186)

        self.cross_attn_video2audio = CrossModalAttentionImproved(feature_dim=128)
        self.cross_attn_audio2video = CrossModalAttentionImproved(feature_dim=128)

        self.temporal_aggregation = HierarchicalTemporalAggregation(feature_dim=256)

        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.fc = SequentialLightweightClassifier(input_dim=256, hidden_dim=128, num_classes=output_dim)
        
    def forward(self, X1, X2):

        X1 = self.video_feature_extractor(X1)
        X1 = self.video_encoder(X1)
        X2 = self.audio_encoder(X2)

        X1_aligned = F.interpolate(
            X1.transpose(1, 2),
            size=X2.size(1),
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        X1_enhanced = self.cross_attn_video2audio(X1_aligned, X2)
        X2_enhanced = self.cross_attn_audio2video(X2, X1_aligned)

        X = torch.cat([X1_enhanced, X2_enhanced], dim=2)

        X = self.temporal_aggregation(X)
        X = self.fusion(X)
        out = self.fc(X)

        return X, out

class BranchNetworkFDFB_nofc(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BranchNetworkFDFB_nofc, self).__init__()     

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, 
            nhead=8, 
            dim_feedforward=1024,
            dropout=0.2,
        )
        self.attn_block = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.global_context = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.Sigmoid()
        )

        self.dynamic_weights = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, X):

        X = self.attn_block(X)

        X_trans = X.permute(0, 2, 1)
        context = self.global_context(X_trans)
        X_trans = X_trans * context
        X_enhanced = X_trans.permute(0, 2, 1)

        X_enhanced = X_enhanced + X

        weights = self.dynamic_weights(X_enhanced)
        weighted_sum = torch.sum(X_enhanced * weights, dim=1)
        
        return X_enhanced, weighted_sum
    
    
class BranchNetworkFDFB_fc(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BranchNetworkFDFB_fc, self).__init__()
        self.sequential_classifier = SequentialLightweightClassifier(
            input_dim=256, 
            hidden_dim=128, 
            num_classes=output_dim
        )
        
    def forward(self, X):
        
        model = self.sequential_classifier(X)
        
        return model
    
    
class BranchNetworkEFB(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BranchNetworkEFB, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=186, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(186)
        self.linear1 = nn.Linear(171, 128)
        self.ln1 = nn.LayerNorm(128)

        self.cross_attn1 = CrossModalAttention(feature_dim=128)
        self.cross_attn2 = CrossModalAttention(feature_dim=128)

        self.attn_block = AttentionBlock(embed_size=hidden_dim, heads=8, num_layers=2, dropout=0.2)

        self.sequential_classifier = SequentialLightweightClassifier(
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim//2, 
            num_classes=output_dim
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, output_dim)
        )
        
    def forward(self, X1, X2):

        X1 = self.conv1(X1)
        X1 = self.bn1(X1)
        X1 = F.gelu(X1)
        X1 = self.linear1(X1)
        X1 = self.ln1(X1)

        X1_enhanced = self.cross_attn1(X1, X2)
        X2_enhanced = self.cross_attn2(X2, X1)

        X = torch.cat([X1_enhanced, X2_enhanced], dim=2)

        X1 = self.attn_block(X)

        X_last = X1[:, -1, :]

        out = self.sequential_classifier(X1)
        
        return X1, X_last, out

class LNet(nn.Module):
    def __init__(self, input_dim1=915, input_dim2=186, hidden_dim=256, output_dim=128):
        super(LNet, self).__init__()
        
        self.branch_NFN = BranchNetworkNFN(input_dim1)
        
        self.branch_DDFB = BranchNetworkDDFB(input_dim1)
        self.branch_FDFB_nofc = BranchNetworkFDFB_nofc(input_dim1, hidden_dim, 2)
        self.branch_FDFB_fc = BranchNetworkFDFB_fc(input_dim1, hidden_dim, 2)
        
        self.branch_EFB = BranchNetworkEFB(input_dim1, hidden_dim, 6)

        self.EGN=DualConsistencyRegularization(seq_len=186, feature_dim=256)
        self.EGN_2=ChildFeatureDynamicRegularization(seq_len=186, feature_dim=256)
        self.TS=AsymmetricDistillation(input_dim=256)
        self.SN=HierarchicalFeatureSeparation(feature_dim=256)
    
        
        self.contrastive_loss = ContrastiveLoss(margin=1.0)
        self.lossFunc = nn.CrossEntropyLoss()
        
    def forward(self, feature1, feature2, batch_size, label=None):

        features_NF_ALL , features_NF = self.branch_NFN(feature1, feature2)

        features_DDF_ALL , output_DDF = self.branch_DDFB(feature1, feature2)

        features_EF_ALL, features_EF, output_EF = self.branch_EFB(feature1,feature2)
        if label is not None:
            label=label.long()
        e_r, e_u, losses=self.SN(features_DDF_ALL, features_NF_ALL, features_EF_ALL, label)
        
        features_DDF_ALL_up, features_NF_ALL_up, features_EF_ALL_up, graph_loss=self.EGN(features_DDF_ALL, features_NF_ALL, features_EF_ALL)
        
        e_r, e_u, loss_flow = self.EGN_2(e_r, e_u)
              
        fused_features, logits, distillation_loss=self.TS(e_r, features_DDF_ALL_up)

        output_DF = self.branch_FDFB_fc(fused_features)

        AC_loss = adversarial_consistency_loss(features_DDF_ALL,features_EF_ALL,features_NF_ALL)

        
        losses_sum = sum(losses.values())

        Loss = AC_loss + (distillation_loss + graph_loss * 0.1 + losses_sum + loss_flow)*0.2
        
        return output_DF, Loss

