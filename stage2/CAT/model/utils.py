import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed

def hard_sample(logits, dim=-1):
    y_soft = F.softmax(logits, dim=-1)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret, index

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var=256):
        super().__init__()
        # actor
        self.obs_layer = nn.Linear(state_dim, n_latent_var)
        self.actor_layer = nn.Sequential(
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim)
        )

    def forward(self, state, action_mask):
        hidden_state = self.obs_layer(state)
        logits = self.actor_layer(hidden_state)
        inf_mask = torch.clamp(torch.log(action_mask.float()),
                               min=torch.finfo(torch.float32).min)
        logits = logits + inf_mask
        actions = hard_sample(logits)
        return actions

class StraightThrough:
    def __init__(self, state_dim, action_dim, lr,  config):
        self.lr = lr
        device = config['device']
        self.betas = config['betas']
        self.policy = Actor(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=self.betas)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

def create_dncoder(config):
    if config['decoder'] == 'ncd':
        return NCDDecoder(config).to(config['device'])

    elif config['decoder'] == 'irt':
        return IRTDecoder(config).to(config['device'])



class NCDDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = Positive_MLP(config).to(config['device'])
        self.transfer_student_layer = nn.Linear(config['out_channels'], config['know_num']).to(config['device'])
        self.transfer_exercise_layer = nn.Linear(config['out_channels'], config['know_num']).to(config['device'])
        self.e_discrimination = nn.Embedding(config['prob_num'], 1)
        self.config = config
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, z, student_id, exercise_id, knowledge_point):
        self.z = z
        state = (torch.sigmoid(self.transfer_student_layer(z[student_id])) - torch.sigmoid(
            self.transfer_exercise_layer(z[self.config['stu_num'] + exercise_id]))) * knowledge_point
        return self.layers.forward(state).view(-1)

    def get_mastery_level(self, z):
        # return torch.sigmoid(z[:self.config['stu_num']]).detach().cpu().numpy()
        return torch.sigmoid(self.transfer_student_layer(z[:self.config['stu_num']])).detach().cpu().numpy()

    def monotonicity(self):
        none_neg_clipper = NoneNegClipper()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)

class IRTDecoder(nn.Module):

    def __init__(
            self, config
    ):
        super().__init__()
        self.theta = nn.Linear(config['out_channels'], config['num_dim'])
        self.alpha = nn.Linear(config['out_channels'], config['num_dim'])
        self.beta = nn.Linear(config['out_channels'], 1)
        self.config = config
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, z, student_id, exercise_id, knowledge_point):
        theta = self.theta(z[student_id])
        alpha = self.alpha(z[self.config['stu_num'] + exercise_id])
        beta = self.beta(z[self.config['stu_num'] + exercise_id])
        pred = (alpha * theta).sum(dim=1, keepdim=True) + beta
        pred = torch.sigmoid(pred)
        return pred.view(-1)

    def get_mastery_level(self, z):
        return torch.sigmoid(z[:self.config['stu_num']]).detach().cpu().numpy()
        # return torch.sigmoid(self.transfer_student_layer(z[:self.config['stu_num']])).detach().cpu().numpy()

    def monotonicity(self):
        pass

class Weighted_Summation(nn.Module):
    def __init__(self, hidden_dim, attn_drop=0.0, dtype=torch.float32):
        super(Weighted_Summation, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True, dtype=dtype)
        nn.init.xavier_normal_(self.fc.weight)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim), dtype=dtype), requires_grad=True)
        nn.init.xavier_normal_(self.att.data)

        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=dtype))

    def forward(self, x):
        # x 的形状是 (N, seq_len, D)
        seq_len, N, D = x.shape
        x = x.permute(1, 0, 2)
        
        # 对输入的每个特征进行线性变换和非线性激活
        transformed = self.tanh(self.fc(x))  # (N, seq_len, D)
        
        # 将注意力参数与变换后的特征进行点积，得到注意力分数
        attn_scores = torch.matmul(transformed, self.att.transpose(0, 1))  # (N, seq_len, 1)
        attn_scores = attn_scores.squeeze(-1)  # (N, seq_len)

        attn_scores = attn_scores / self.scale
        
        # 对注意力分数应用 softmax
        self.attn_scores = self.softmax(attn_scores)  # (N, seq_len)
        # print(self.attn_scores[0])
        # torch.save(self.attn_scores, 'attention_matrix.pt')
        
        # 如果指定了 dropout，则应用 dropout
        self.attn_scores = self.attn_drop(attn_scores)  # (N, seq_len)
        
        # 使用注意力权重对原始输入进行加权求和
        weighted_sum = torch.bmm(attn_scores.unsqueeze(1), x).squeeze(1)  # (N, D)
        
        return weighted_sum


def get_mlp_encoder(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, 512),
        nn.PReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.PReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, out_channels),
    )

def Positive_MLP(config, num_layers=3, hidden_dim=512, dropout=0.5):
    layers = []
    layers.append(nn.Linear(config['know_num'], 128))
    layers.append(nn.Dropout(p=0.5))
    layers.append(nn.Linear(128, 64))
    layers.append(nn.Dropout(p=0.5))
    layers.append(nn.Linear(64, 1))
    layers.append(nn.Sigmoid())
    layers = nn.Sequential(*layers)
    return layers

def calculate_doa_k_block(mas_level, q_matrix, r_matrix, k, block_size=50):
    n_students, n_skills = mas_level.shape
    n_questions, _ = q_matrix.shape
    numerator = 0
    denominator = 0
    question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
    for start in range(0, n_students, block_size):
        end = min(start + block_size, n_students)
        delta_matrix_block = mas_level[start:end, k].reshape(-1, 1) > mas_level[start:end, k].reshape(1, -1)
        r_matrix_block = r_matrix[start:end, :]
        for j in question_hask:
            row_vector = (r_matrix_block[:, j].reshape(1, -1) != -1).astype(int)
            columen_vector = (r_matrix_block[:, j].reshape(-1, 1) != -1).astype(int)
            mask = row_vector * columen_vector
            delta_r_matrix = r_matrix_block[:, j].reshape(-1, 1) > r_matrix_block[:, j].reshape(1, -1)
            I_matrix = r_matrix_block[:, j].reshape(-1, 1) != r_matrix_block[:, j].reshape(1, -1)
            numerator_ = np.logical_and(mask, delta_r_matrix)
            denominator_ = np.logical_and(mask, I_matrix)
            numerator += np.sum(delta_matrix_block * numerator_)
            denominator += np.sum(delta_matrix_block * denominator_)
    if denominator == 0:
        DOA_k = 0
    else:
        DOA_k = numerator / denominator
    return DOA_k

def get_top_k_concepts(q, a, topk: int = 10):
    # q = pd.read_csv('../data/{}/q.csv'.format(datatype), header=None).to_numpy()
    # a = pd.read_csv('../data/{}/TotalData.csv'.format(datatype), header=None).to_numpy()
    skill_dict = {}
    for k in range(q.shape[1]):
        skill_dict[k] = 0
    for k in range(a.shape[0]):
        stu_id = a[k, 0]
        prob_id = a[k, 1]
        skills = np.where(q[int(prob_id), :] != 0)[0].tolist()
        for skill in skills:
            skill_dict[skill] += 1

    sorted_dict = dict(sorted(skill_dict.items(), key=lambda x: x[1], reverse=True))
    all_list = list(sorted_dict.keys())  # 189
    return all_list[:topk]

def get_doa(config, mastery_level):
    q_matrix = config['q'].to_numpy()
    r_matrix = config['r']
    check_concepts = get_top_k_concepts(q_matrix, config['train_triplets'])
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in check_concepts)
    return np.mean(doa_k_list)

def get_r_matrix(np_test, stu_num, prob_num, new_idx=None):
    if new_idx is None:
        r = -1 * np.ones(shape=(stu_num, prob_num))
        for i in range(np_test.shape[0]):
            s = int(np_test[i, 0])
            p = int(np_test[i, 1])
            score = np_test[i, 2]
            r[s, p] = int(score)
    else:
        r = -1 * np.ones(shape=(stu_num, prob_num))

        for i in range(np_test.shape[0]):
            s = new_idx.index(int(np_test[i, 0]))
            p = int(np_test[i, 1])
            score = np_test[i, 2]
            r[s, p] = int(score)
    return r