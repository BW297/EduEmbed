import torch
import torch.nn as nn
import torch.nn.functional as F

def get_mlp(input_channel, output_channel, dtype, device):
    return nn.Sequential(
        nn.Linear(input_channel, 512, dtype=dtype).to(device),
        nn.PReLU(device=device, dtype=dtype),
        nn.Dropout(0.5),
        nn.Linear(512, 256, dtype=dtype).to(device),
        nn.PReLU(device=device, dtype=dtype),
        nn.Dropout(0.5),
        nn.Linear(256, output_channel, dtype=dtype).to(device)
    )
class PosLinear(nn.Linear):  # 确保权重非负
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight  # relu=max(0,x)
        return F.linear(input, weight, self.bias)

def get_stu_mlp(input_channel, output_channel, dtype, device):
    return nn.Sequential(
        PosLinear(input_channel, 512, dtype=dtype).to(device),
        nn.PReLU(device=device, dtype=dtype),
        nn.Dropout(0.5),
        PosLinear(512, 256, dtype=dtype).to(device),
        nn.PReLU(device=device, dtype=dtype),
        nn.Dropout(0.5),
        PosLinear(256, output_channel, dtype=dtype).to(device)
    )

class extract_fea(nn.Module):
    def __init__(self, dataset, latent_dim, dtype, device):
        super(extract_fea, self).__init__()
        self.latent_dim = latent_dim
        self.dtype = dtype
        self.device = device
        self.emb_map = {}
        _,_,_,_, self.known, _ = dataset.get_num()

        train_emb, test_emb = dataset.get_text_embedding(dtype, device)
        self.dict = {
            "train": train_emb,
            "test": test_emb
        }
        self.text_dim = self.dict["train"]["student"].shape[1]
        self.know = get_mlp(self.text_dim, self.latent_dim, dtype, device)
        self.exer = get_mlp(self.text_dim, self.latent_dim, dtype, device)
        self.stu = get_mlp(self.text_dim, self.latent_dim, dtype, device)
        self.disc = get_mlp(self.text_dim, 1, dtype, device)
        self.emb = ...
        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def extract(self, student_id, exercise_id, type):
        if type == "train":
            self.emb = self.dict["train"]
        else:
            self.emb = self.dict["test"]
        student_ts = self.stu(self.emb["student"])[student_id]
        disc_ts = self.disc(self.emb["disc"])[exercise_id]
        diff_ts = self.exer(self.emb["exercise"])[exercise_id]
        knowledge_ts = self.know(self.emb["knowledge"])
        return student_ts, diff_ts, disc_ts, knowledge_ts

    def update(self, item):
        self.eval()
        self.emb_map["student"] = self.stu(self.emb["student"])
        self.emb_map["knowledge"] = self.know(self.emb["knowledge"])
        return self.emb_map[item]