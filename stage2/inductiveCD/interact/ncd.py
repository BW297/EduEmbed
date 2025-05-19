import torch
import torch.nn as nn
import torch.nn.functional as F

class PosLinear(nn.Linear):  # 确保权重非负
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight  # relu=max(0,x)
        return F.linear(input, weight, self.bias)

class NCD_IF(nn.Module):
    def __init__(self, know_n,device, dtype):
        super(NCD_IF, self).__init__()
        self.knowledge_num = know_n
        self.device = device
        self.dtype = dtype

        self.k_diff_full = PosLinear(self.knowledge_num, 1, dtype=dtype).to(self.device)
        self.stat_full = PosLinear(self.knowledge_num, 1, dtype=dtype).to(self.device)

        self.score_mlp = nn.Sequential(
            PosLinear(self.knowledge_num, 512, dtype=dtype).to(device),
            nn.Tanh(),
            nn.Dropout(0.5),
            PosLinear(512, 256, dtype=dtype).to(device),
            nn.Tanh(),
            nn.Dropout(0.5),
            PosLinear(256, 1, dtype=dtype).to(device),
            nn.Sigmoid()
        )
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def compute(self, student_ts, diff_ts, disc_ts, knowledge_ts, q_mask):
        input_x = torch.sigmoid(disc_ts) * (torch.sigmoid(student_ts)
                                             - torch.sigmoid(diff_ts)) * q_mask
        return self.score_mlp(input_x).view(-1)

    def transform(self, mastery, knowledge):
        self.eval()
        print(mastery.shape)
        return torch.sigmoid(mastery)
