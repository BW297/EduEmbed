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
class extract_fea(nn.Module):
    def __init__(self, dataset, latent_dim, dtype, device,lamda,alpha):
        super(extract_fea, self).__init__()
        self.latent_dim = latent_dim
        self.dtype = dtype
        self.device = device
        self.lam=lamda
        self.alpha=alpha
        self.emb_map = {}

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

        _,_,user_n,exer_n,tr_know_n,to_know_n=dataset.get_num()
        know_n=tr_know_n
        self.know_emb = nn.Parameter(torch.zeros(know_n, latent_dim,dtype=dtype)).to(device)
        self.exer_emb=nn.Embedding(exer_n,latent_dim,dtype=dtype).to(device)
        self.stu_emb=nn.Embedding(user_n,latent_dim,dtype=dtype).to(device)
        self.disc_emb =nn.Embedding(exer_n,1,dtype=dtype).to(device)
        
        self.know_concat=nn.Linear(2*self.latent_dim,latent_dim,dtype=dtype).to(device)
        self.exer_concat=nn.Linear(2*self.latent_dim,latent_dim,dtype=dtype).to(device)
        self.stu_concat=nn.Linear(2*self.latent_dim,latent_dim,dtype=dtype).to(device)
        self.disc_concat =nn.Linear(2,1,dtype=dtype).to(device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.know_emb)
    def extract(self, student_id, exercise_id, type):

        if type == "train":
            self.emb = self.dict["train"]
        else:
            self.emb = self.dict["test"]

        student_ts1 = self.stu(self.emb["student"])[student_id]
        disc_ts1 = self.disc(self.emb["disc"])[exercise_id]
        diff_ts1 = self.exer(self.emb["exercise"])[exercise_id]
        knowledge_ts1 = self.know(self.emb["knowledge"])

        student_ts2 = self.stu_emb(student_id)
        disc_ts2 = self.disc_emb(exercise_id)
        diff_ts2 = self.exer_emb(exercise_id)
        knowledge_ts2 = self.know_emb

        student_ts=self.lam*student_ts1+(1-self.lam)*student_ts2
        disc_ts=self.lam*disc_ts1+(1-self.lam)*disc_ts2
        diff_ts=self.lam*diff_ts1+(1-self.lam)*diff_ts2
        knowledge_ts=self.lam*(knowledge_ts1)+(1-self.lam)*knowledge_ts2
        
        return student_ts, diff_ts, disc_ts, knowledge_ts,{"extra_loss":self.cal_loss(student_id,exercise_id)}

    def update(self, item):
        self.eval()
        student1=self.stu(self.emb["student"])
        student2=self.stu_emb.weight
        know1=self.know(self.emb["knowledge"])
        know2=self.know_emb
        self.emb_map["student"] = self.lam*student1+(1-self.lam)*student2
        self.emb_map["knowledge"] = self.lam*know1+(1-self.lam)*know2
        return self.emb_map[item]
    
    def get_id_and_get_text(self,student_id, exercise_id):
        stu_text=self.stu(self.emb['student'])[student_id]
        stu_id=self.stu_emb(student_id)
        exer_text=self.exer(self.emb['exercise'])[exercise_id]
        exer_id=self.exer_emb(exercise_id)
        know_text=self.know(self.emb['knowledge'])
        know_id=self.know_emb
        disc_text=self.disc(self.emb['disc'])[exercise_id]
        disc_id=self.disc_emb(exercise_id)
        set={
            'stu':[stu_text,stu_id],
            'exer':[exer_text,exer_id],
            'disc':[disc_text,disc_id],
            'know':[know_text,know_id]
        }
        return set
        
    def ssl_con_loss(self,x_y, temp=1.0):
        x,y=x_y
        x = F.normalize(x)
        y = F.normalize(y)
        sim_matrix = x @ y.T / temp
        batch_size = x.size(0)
        mask = torch.eye(batch_size, device=x.device).bool()
    
        # 用一个很小的数掩盖对角线，避免正样本进入deno
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        mole = torch.exp(torch.sum(x * y, dim=1) / temp)
        deno = torch.sum(torch.exp(sim_matrix), dim=1)
        loss = -torch.log(mole / (deno + 1e-8) + 1e-8).mean()
        return loss
    
    def cal_loss(self,student_id,exercise_id):
        data=self.get_id_and_get_text(student_id,exercise_id)
        loss=0
        loss+=self.ssl_con_loss(data['stu'])
        loss+=self.ssl_con_loss(data['exer'])
        loss+=self.ssl_con_loss(data['disc'])
        loss+=self.ssl_con_loss(data['know'])
        return loss*self.alpha
        
