import sys
sys.path.append('..')
import argparse
import os
import torch
import json
import random
from model.utils import StraightThrough
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.utils import data
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def initialize_seeds(seedNum):
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    random.seed(seedNum)

def data_split(datapath, fold, seed, n_student):
    response_logs = pd.read_csv(datapath, header=None, names=['user_id', 'q_id', 'label'])
    data = []
    for stu in range(n_student):
        tmp ={'user_id': int(stu), 'q_ids': [], 'labels': []}
        logs = response_logs.loc[response_logs['user_id'] == stu]
        for log in logs.values:
            tmp['q_ids'].append(int(log[1]))
            tmp['labels'].append(int(log[2]))
        data.append(tmp)
    # random.Random(seed).shuffle(data)
    fields = ['q_ids',  'labels']  # 'ans', 'correct_ans',
    del_fields = []
    for d in data:
        for f in fields:
            d[f] = np.array(d[f])
    # print(data)
    total_stu = list(range(0, n_student))
    train_stu = random.sample(total_stu, int(len(total_stu) * 0.2))
    train_data = [stu for stu in data if stu['user_id'] in train_stu]
    return train_data

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data, seed=None):
        'Initialization'
        self.data = data
        self.seed = seed

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        # return self.data[index]
        'Generates one sample of data'
        random.seed(0)
        data = self.data[index]
        observed_index = np.array([idx for idx in range(len(data['q_ids']))])
        trainable_index = random.sample(observed_index.tolist(), int(len(observed_index) * 0.8))
        target_index = [index for index in observed_index if index not in trainable_index]

        # input_ans = data['ans'][trainable_index]
        input_label = data['labels'][trainable_index]
        input_question = data['q_ids'][trainable_index]
        output_label = data['labels'][target_index]
        output_question = data['q_ids'][target_index]

        output = {'input_label': torch.FloatTensor(input_label), 'input_question': torch.FloatTensor(input_question),
                  'output_question': torch.FloatTensor(output_question), 'output_label': torch.FloatTensor(output_label)}
        # 'input_ans': torch.FloatTensor(input_ans)
        return output


class collate_fn(object):
    def __init__(self, n_question):
        self.n_question = n_question

    def __call__(self, batch):
        B = len(batch)
        input_labels = torch.zeros(B, self.n_question).long()
        output_labels = torch.zeros(B, self.n_question).long()
        #input_ans = torch.ones(B, self.n_question).long()
        input_mask = torch.zeros(B, self.n_question).long()
        output_mask = torch.zeros(B, self.n_question).long()
        for b_idx in range(B):
            input_labels[b_idx, batch[b_idx]['input_question'].long(
            )] = batch[b_idx]['input_label'].long()
            #input_ans[b_idx, batch[b_idx]['input_question'].long()] = batch[b_idx]['input_ans'].long()
            input_mask[b_idx, batch[b_idx]['input_question'].long()] = 1
            output_labels[b_idx, batch[b_idx]['output_question'].long(
            )] = batch[b_idx]['output_label'].long()
            output_mask[b_idx, batch[b_idx]['output_question'].long()] = 1

        output = {'input_labels': input_labels,  'input_mask': input_mask,
                  'output_labels': output_labels, 'output_mask': output_mask}
        # 'input_ans':input_ans,
        return output

def get_inputs(batch):
    input_labels = batch['input_labels'].to(device).float()
    input_mask = batch['input_mask'].to(device)
    #input_ans = batch['input_ans'].to(device)-1
    input_ans = None
    return input_labels, input_ans, input_mask

def get_outputs(batch):
    output_labels, output_mask = batch['output_labels'].to(
        device).float(), batch['output_mask'].to(device)  # B,948
    return output_labels, output_mask

def compute_loss(output, labels, mask, reduction= True):

    loss_function = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_function(output, labels) * mask
    if reduction:
        return loss.sum()/mask.sum()
    else:
        return loss.sum()

def normalize_loss(output, labels, mask):
    loss_function = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_function(output, labels) * mask
    count = mask.sum(dim =-1)+1e-8#N,1
    loss = 10. * torch.sum(loss, dim =-1)/count
    return loss.sum()

class MAMLModel(nn.Module):
    def __init__(self, n_question,question_dim =1,dropout=0.2, sampling='active', n_query=10,emb = None,tp='irt'):
        super().__init__()
        self.n_query = n_query
        self.sampling = sampling
        self.sigmoid = nn.Sigmoid()
        self.n_question = n_question
        self.question_dim = question_dim
        self.tp = tp
        if tp == 'irt':
            self.question_difficulty = nn.Parameter(torch.zeros(question_dim,n_question))     
        elif tp == 'ncd':
            self.prednet_input_len = emb.shape[1]
            self.prednet_len1, self.prednet_len2 = 128, 64  # changeable
            self.kn_emb = emb
            #self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
            self.k_difficulty = nn.Parameter(torch.zeros(n_question,self.prednet_input_len))
            self.e_discrimination = nn.Parameter(torch.full((n_question,1), 0.5))
            self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
            self.drop_1 = nn.Dropout(p=0.5)
            self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
            self.drop_2 = nn.Dropout(p=0.5)
            self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        
    def reset(self, batch):
        input_labels, _, input_mask = get_inputs(batch)
        obs_state = ((input_labels-0.5)*2.)  # B, 948
        train_mask = torch.zeros(
            input_mask.shape[0], self.n_question).long().to(device)
        env_states = {'obs_state': obs_state, 'train_mask': train_mask,
                      'action_mask': input_mask.clone()}
        return env_states
    

    def step(self, env_states):
        obs_state,  train_mask = env_states[
            'obs_state'], env_states['train_mask']
        state = obs_state*train_mask  # B, 948
        return state

    def pick_sample(self,sampling, config):
        student_embed = config['meta_param']
        n_student = len(config['meta_param'])
        action = self.pick_uncertain_sample(student_embed, config['available_mask'])
        config['train_mask'][range(n_student), action], config['available_mask'][range(n_student), action] = 1, 0
        return action
        

    def forward(self, batch, config):
        #get inputs
        input_labels = batch['input_labels'].to(device).float()
        student_embed = config['meta_param']#
        output = self.compute_output(student_embed)
        train_mask = config['train_mask']
        #compute loss
        if config['mode'] == 'train':
            output_labels, output_mask = get_outputs(batch)
            #meta model parameters 
            output_loss = compute_loss(output, output_labels, output_mask, reduction=False)/len(train_mask)
            #for adapting meta model parameters
            if self.n_query!=-1:
                input_loss = compute_loss(output, input_labels, train_mask, reduction=False)
            else:
                input_loss = normalize_loss(output, input_labels, train_mask)
            #loss = input_loss*self.alpha + output_loss
            return {'loss': output_loss, 'train_loss': input_loss, 'output': self.sigmoid(output).detach().cpu().numpy()}
        else:
            input_loss = compute_loss(output, input_labels, train_mask,reduction=False)
            return {'output': self.sigmoid(output).detach().cpu().numpy(), 'train_loss': input_loss}

    def pick_uncertain_sample(self, student_embed, available_mask):
        with torch.no_grad():
            output = self.compute_output(student_embed)
            output = self.sigmoid(output)
            inf_mask = torch.clamp(
                torch.log(available_mask.float()), min=torch.finfo(torch.float32).min)
            scores = torch.min(1-output, output)+inf_mask
            actions = torch.argmax(scores, dim=-1)
            return actions

    def compute_output(self, student_embed):
        if self.tp=='irt':
            output = (student_embed - self.question_difficulty)
        else:
            k_difficulty = self.k_difficulty
            e_discrimination = self.e_discrimination
            kn_emb = self.kn_emb
            # prednet
            student_embed = student_embed.unsqueeze(1)
            input_x = e_discrimination * (student_embed - k_difficulty) *kn_emb.to(device)
            input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
            input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
            output = self.prednet_full3(input_x)
            output = output.squeeze()
        return output
        
def clone_meta_params(batch):
    return [meta_params[0].expand(len(batch['input_labels']),  -1).clone()]

def inner_algo(batch, config, new_params, create_graph=False):

    for _ in range(params.inner_loop):
        config['meta_param'] = new_params[0]
        res = model(batch, config)
        loss = res['train_loss']
        grads = torch.autograd.grad(
            loss, new_params, create_graph=create_graph)
        new_params = [(new_params[i] - params.inner_lr*grads[i])
                      for i in range(len(new_params))]
        del grads
    config['meta_param'] = new_params[0]
    return

def run_biased(batch, config):
    new_params = clone_meta_params(batch)
    if config['mode'] == 'train':
        model.eval()
    pick_biased_samples(batch, config)
    optimizer.zero_grad()
    meta_params_optimizer.zero_grad()
    inner_algo(batch, config, new_params)
    if config['mode'] == 'train':
        model.train()
        optimizer.zero_grad()
        res = model(batch, config)
        loss = res['loss']
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
    else:
        with torch.no_grad():
            res = model(batch, config)

    return res['output']

def pick_biased_samples(batch, config):
    new_params = clone_meta_params(batch)
    env_states = model.reset(batch)
    action_mask, train_mask = env_states['action_mask'], env_states['train_mask']
    for i in range(params.n_query):
        with torch.no_grad():
            state = model.step(env_states)
            train_mask = env_states['train_mask']
        if config['mode'] == 'train':
            train_mask_sample, actions = st_policy.policy(state, action_mask)
        else:
            with torch.no_grad():
                train_mask_sample, actions = st_policy.policy(
                    state, action_mask)
        action_mask[range(len(action_mask)), actions] = 0
        # env state train mask should be detached
        env_states['train_mask'], env_states['action_mask'] = train_mask + \
            train_mask_sample.data, action_mask
        if config['mode'] == 'train':
            # loss computation train mask should flow gradient
            config['train_mask'] = train_mask_sample+train_mask
            inner_algo(batch, config, new_params, create_graph=True)
            res = model(batch, config)
            loss = res['loss']
            st_policy.update(loss)

    config['train_mask'] = env_states['train_mask']
    return 

def create_parser():
    parser = argparse.ArgumentParser(description='ML')
    parser.add_argument('--model', type=str,
                        default='biirt-biased', help='type')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='type')
    parser.add_argument('--lr', type=float, default=1e-4, help='type') #
    parser.add_argument('--meta_lr', type=float, default=1e-4, help='type')
    parser.add_argument('--inner_lr', type=float, default=1e-1, help='type') #
    parser.add_argument('--inner_loop', type=int, default=5, help='type') #
    parser.add_argument('--policy_lr', type=float, default=2e-3, help='type') #
    parser.add_argument('--dropout', type=float, default=0.6, help='type')
    parser.add_argument('--dataset', type=str,
                        default='FrcSub', help='NeurIPS2020 , XES3G5M or MOOCRadar')
    parser.add_argument('--fold', type=int, default=1, help='type')
    parser.add_argument('--n_query', type=int, default=20, help='type')
    parser.add_argument('--seed', type=int, default=0, help='type')
    parser.add_argument('--device', type=str, default='cuda:0')

    params = parser.parse_args()


    if params.dataset == 'SLP-CHIsp':
        params.n_student = 562
        params.n_question = 510
        params.concept_num = 17
        params.train_batch_size = 32
        params.test_batch_size = 32
        params.n_epoch = 100
    elif params.dataset == 'SLP-MAT':
        params.n_student = 1080
        params.n_question = 609
        params.concept_num = 32
        params.train_batch_size = 32
        params.test_batch_size = 32
        params.n_epoch = 100
    return params


def train_model():
    config['mode'] = 'train'
    config['epoch'] = epoch
    model.train()
    for batch in train_loader:
        # Select RL Actions, save in config
        run_biased(batch, config)

    #   
if __name__ == "__main__":
    params = create_parser()
    print(params)
    config = {
        'policy_path': '../model/ckpt/{}_policy.pt'.format(params.model.split('-')[0]),
        'betas': (0.9, 0.999),
        'device':params.device
    }
    initialize_seeds(params.seed)
    device = params.device

    base, sampling = params.model.split('-')[0], params.model.split('-')[-1]
    if base == 'biirt':
        model = MAMLModel(sampling=sampling, n_query=params.n_query,
                          n_question=params.n_question, question_dim=1,tp = 'irt').to(device)
        meta_params = [torch.zeros(1, 1, device=device, requires_grad=True)]

    if base == 'binn':
        q = pd.read_csv(f'../../datasets/{params.dataset}/q_matrix.csv', header=None)
        concepts = {}
        cnt = 0
        for question in q.values:
            concepts[cnt] = np.where(np.array(question))[0].tolist()
            cnt += 1
        num_concepts = params.concept_num
        concepts_emb = [[0.] * num_concepts for i in range(params.n_question)]
        for i in range(params.n_question):
            for concept in concepts[i]:
                concepts_emb[i][concept] = 1.0
        concepts_emb = torch.tensor(concepts_emb, dtype=torch.float32).to(device)
        model = MAMLModel(sampling=sampling, n_query=params.n_query,
                          n_question=params.n_question, question_dim=num_concepts,tp ='ncd',emb=concepts_emb).to(device)
        meta_params = [torch.zeros((1, num_concepts), device=device, requires_grad=True)]

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=1e-8)
    meta_params_optimizer = torch.optim.SGD(
        meta_params, lr=params.meta_lr, weight_decay=2e-6, momentum=0.9)
    st_policy = StraightThrough(params.n_question, params.n_question,
                                params.policy_lr, config)

    data_path = os.path.normpath('../../datasets/'+params.dataset+'/response.csv')
    train_data = data_split(data_path, params.fold,  params.seed, params.n_student)
    train_dataset = Dataset(train_data)

    num_workers = 3
    collate_fn = collate_fn(params.n_question)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=params.train_batch_size, shuffle=True, drop_last=False)
    for epoch in tqdm(range(params.n_epoch)):
        train_model()
    torch.save(st_policy.policy.state_dict(),config['policy_path'])
