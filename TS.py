import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricDistillation(nn.Module):
    def __init__(self, input_dim=256):
        super(AsymmetricDistillation, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim_mlp1 = 128
        self.hidden_dim_mlp2 = 64
        self.teacher_dim = 128
        self.student_dim = 64

        self.norm_edep = nn.LayerNorm(input_dim)
        self.norm_d = nn.LayerNorm(input_dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim_mlp1),
            nn.LayerNorm(self.hidden_dim_mlp1),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim_mlp2),
            nn.LayerNorm(self.hidden_dim_mlp2),
            nn.ReLU()
        )
        self.mlp1_to_mlp2_align = nn.Sequential(
            nn.Linear(self.hidden_dim_mlp1, self.hidden_dim_mlp2),
            nn.LayerNorm(self.hidden_dim_mlp2)
        )

        self.linear_gate = nn.Linear(input_dim, 1)

        self.teacher_gru1 = nn.GRU(input_dim, input_dim, batch_first=True)
        self.norm_teacher1 = nn.LayerNorm(input_dim)
        
        self.teacher_gru2 = nn.GRU(input_dim, input_dim, batch_first=True)
        self.norm_teacher2 = nn.LayerNorm(input_dim)
        
        self.teacher_gru3 = nn.GRU(input_dim, self.teacher_dim, batch_first=True)
        self.norm_teacher3 = nn.LayerNorm(self.teacher_dim)
        
        self.student_gru = nn.GRU(input_dim, self.student_dim, batch_first=True)
        self.norm_student = nn.LayerNorm(self.student_dim)
        
        self.linear_projection = nn.Sequential(
            nn.Linear(self.student_dim, self.teacher_dim),
            nn.LayerNorm(self.teacher_dim)
        )

        self.align_distilled = nn.Sequential(
            nn.Linear(self.student_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        self.norm_fused = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, 2)

    def asymmetric_refinement(self, f_edep, f_d):

        f_edep = self.norm_edep(f_edep)
        f_d = self.norm_d(f_d)
        
        batch_size, seq_len, _ = f_edep.shape

        f_edep_reshaped = f_edep.reshape(-1, self.input_dim)
        f_d_reshaped = f_d.reshape(-1, self.input_dim)

        edep_encoded = self.mlp1(f_edep_reshaped)
        d_encoded = self.mlp2(f_d_reshaped)

        edep_mapped = self.mlp1_to_mlp2_align(edep_encoded)

        attention_score = torch.sum(edep_mapped * d_encoded, dim=1) / (self.hidden_dim_mlp2 ** 0.5)
        attention_score = attention_score.reshape(batch_size, seq_len)

        diff = f_edep_reshaped - f_d_reshaped
        gate = torch.sigmoid(self.linear_gate(diff)).squeeze(-1).reshape(batch_size, seq_len)

        attention_weights = attention_score * gate

        attention_weights = F.softmax(attention_weights, dim=1)

        refined_features = f_edep * attention_weights.unsqueeze(-1)
        
        return refined_features, attention_weights

    def feature_distillation(self, refined_features):

        with torch.cuda.amp.autocast(enabled=True):
            out1, _ = self.teacher_gru1(refined_features)
            out1 = self.norm_teacher1(out1)
            
            out2, _ = self.teacher_gru2(out1)
            out2 = self.norm_teacher2(out2)
            
            teacher_output, _ = self.teacher_gru3(out2)
            teacher_output = self.norm_teacher3(teacher_output)

            student_output, _ = self.student_gru(refined_features)
            student_output = self.norm_student(student_output)

            student_projected = self.linear_projection(student_output)

            temperature = 2.0

            teacher_probs = F.softmax(teacher_output / temperature, dim=2)
            log_student_probs = F.log_softmax(student_projected / temperature, dim=2)

            kl_div = F.kl_div(
                log_student_probs.reshape(-1, self.teacher_dim), 
                teacher_probs.reshape(-1, self.teacher_dim),
                reduction='batchmean'
            ) * (temperature ** 2)

            mse_loss = F.mse_loss(student_projected, teacher_output)
            x=0.114;
            distillation_loss = kl_div*x +  mse_loss
        
        return student_output, distillation_loss

    def fusion_and_classification(self, f_edep_dist, f_d):

        f_d = self.norm_d(f_d)

        f_edep_aligned = self.align_distilled(f_edep_dist)

        fused_features = f_d + 0.2 * f_edep_aligned
        fused_features = self.norm_fused(fused_features)

        logits = self.classifier(fused_features)
        return fused_features, logits

    def forward(self, f_edep, f_d):

        f_edep_refined, attention_weights = self.asymmetric_refinement(f_edep, f_d)
        f_edep_dist, distillation_loss = self.feature_distillation(f_edep_refined)
        fused_features, logits = self.fusion_and_classification(f_edep_dist, f_d)
        return fused_features, logits, distillation_loss
