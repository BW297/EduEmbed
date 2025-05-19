import torch
import torch.nn.functional as F
import logging
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,accuracy_score,mean_squared_error, f1_score
import math
from model.abstract_model import AbstractModel
from dataset import AdapTestDataset, TrainDataset, Dataset
from .utils import StraightThrough, get_doa, NCDDecoder
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
class NCD(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, config, student_n, exer_n, knowledge_n, prednet_len1=128, prednet_len2=64):
        self.config = config
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = prednet_len1, prednet_len2  # changeable
        self.lam=config['lamda']

        

        super(NCD, self).__init__()
        self.decoder = NCDDecoder(config)

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, config['out_channels'])
        self.k_difficulty = nn.Embedding(self.exer_n, config['out_channels'])
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        self.init_stu = torch.tensor(self.config['LLM_stu_emb'], dtype=torch.float32).to(config['device'])
        self.init_exer = torch.tensor(self.config['LLM_exer_emb'], dtype=torch.float32).to(config['device'])
        self.init_know = torch.tensor(self.config['LLM_know_emb'], dtype=torch.float32).to(config['device'])
            
        self.emb_num = self.init_exer.shape[1]
        
        self.LLM_transfer_stu =get_mlp(self.emb_num, config['out_channels'],dtype=torch.float32,device=config['device'])
        self.LLM_transfer_k_difficulty = get_mlp(self.emb_num, config['out_channels'],dtype=torch.float32,device=config['device'])
        self.LLM_transfer_e_discrimination = get_mlp(self.emb_num, 1,dtype=torch.float32,device=config['device'])

    
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape)>1:
                    nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        student_factor=self.student_emb.weight
        exercise_factor=self.lam*self.LLM_transfer_k_difficulty(self.init_exer)+(1-self.lam)*self.k_difficulty.weight
        final_x = torch.cat([student_factor, exercise_factor], dim=0)
        return self.decoder.forward(final_x, stu_id, exer_id, kn_emb)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)
    
    def get_mastery_level(self):
        id=self.student_emb.weight.data.detach().cpu()
        return torch.sigmoid(id).numpy()


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class NCDModel(AbstractModel):

    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.model = None
        self.alpha=config['alpha']

    @property
    def name(self):
        return 'Neural Cognitive Diagnosis'

    def init_model(self, data: Dataset):
        policy_lr=0.0005
        self.model = NCD(self.config, data.num_students, data.num_questions, data.num_concepts, self.config['prednet_len1'], self.config['prednet_len2'])
        self.policy = StraightThrough(data.num_questions, data.num_questions, policy_lr, self.config)
        self.n_s = data.num_students
        self.n_q = data.num_questions
        self.model.to(self.config['device'])
    
    def train(self, train_data: TrainDataset):
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        self.model.to(device)
        logging.info('train on {}'.format(device))

        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for ep in range(1, epochs + 1):
            loss = 0.0
            log_step = 1
            epoch_losses = []
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(tqdm(train_loader, "Epoch %s" % ep)):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                concepts_emb = concepts_emb.to(device)
                labels = labels.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)+self.cal_loss(student_ids,question_ids)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.apply_clipper()
                loss += bz_loss.data.float()
                epoch_losses.append(loss.item())
            print(f'[{ep:03d}/{epochs}] | Loss: {np.mean(epoch_losses):.4f}')
    
    def _loss_function(self, pred, real):
        return -(real * torch.log(0.0001 + pred) + (1 - real) * torch.log(1.0001 - pred)).mean()
    
    def adaptest_save(self, path):
        """
        Save the model. Do not save the parameters for students.
        """
        model_dict = self.model.state_dict()
        model_dict = {k:v for k,v in model_dict.items() if 'student' not in k}
        torch.save(model_dict, path)
    
    def adaptest_load(self, path):
        """
        Reload the saved model
        """
        if self.config['policy'] =='bobcat':
            self.policy.policy.load_state_dict(torch.load(self.config['policy_path']),strict=False)
        self.model.load_state_dict(torch.load(path,map_location="cuda:0"), strict=False)
        self.model.to(self.config['device'])
    
    def adaptest_update(self, adaptest_data: AdapTestDataset,sid=None):
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        tested_dataset = adaptest_data.get_tested_dataset(last=True,ssid=sid)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)

        for ep in range(1, epochs + 1):
            loss = 0.0
            log_steps = 1
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(dataloader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device)
                concepts_emb = concepts_emb.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)+self.cal_loss(student_ids,question_ids)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.apply_clipper()
                loss += bz_loss.data.float()
        return loss

    def evaluate(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        real = []
        pred = []
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                real += [data[sid][qid] for qid in question_ids]
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids, concepts_embs).view(-1)
                pred += output.tolist()
            self.model.train()

        coverages = []
        for sid in data:
            all_concepts = set()
            tested_concepts = set()
            for qid in data[sid]:
                all_concepts.update(set(concept_map[qid]))
            for qid in adaptest_data.tested[sid]:
                tested_concepts.update(set(concept_map[qid]))
            coverage = len(tested_concepts) / len(all_concepts)
            coverages.append(coverage)
        cov = sum(coverages) / len(coverages)

        real = np.array(real)
        pred = np.array(pred)
        auc = roc_auc_score(real, pred)
        rmse = np.sqrt(mean_squared_error(real, pred))

        
        # Calculate accuracy
        threshold = 0.5  # You may adjust the threshold based on your use case
        binary_pred = (pred >= threshold).astype(int)
        acc = accuracy_score(real, binary_pred)
        f1 = f1_score(real, binary_pred)
        doa = get_doa(self.config, self.model.decoder.get_mastery_level(self.model.decoder.z))

        return {
            'auc': auc,
            'acc': acc,
            'doa': doa
        }
    
    def get_pred(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        pred_all = {}
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                pred_all[sid] = {}
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids, concepts_embs).view(-1).tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.model.train()
        return pred_all

    def expected_model_change(self, sid: int, qid: int, adaptest_data: AdapTestDataset, pred_all: dict):
        """ get expected model change
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            float, expected model change
        """
        epochs = self.config['num_epochs']
        lr = self.config['learning_rate']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for name, param in self.model.named_parameters():
            if 'student' not in name:
                param.requires_grad = False

        original_weights = self.model.student_emb.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        concepts = adaptest_data.concept_map[qid]
        concepts_emb = [0.] * adaptest_data.num_concepts
        for concept in concepts:
            concepts_emb[concept] = 1.0
        concepts_emb = torch.Tensor([concepts_emb]).to(device)
        correct = torch.LongTensor([1]).to(device)
        wrong = torch.LongTensor([0]).to(device)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb)
            loss = self._loss_function(pred, correct)
            loss.backward()
            optimizer.step()
            
        pos_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb)
            loss = self._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        for param in self.model.parameters():
            param.requires_grad = True

        pred = pred_all[sid][qid]
        return pred * torch.norm(pos_weights - original_weights).item() + \
               (1 - pred) * torch.norm(neg_weights - original_weights).item()
               
    def get_BE_weights(self, pred_all):
        """
        Returns:
            predictions, dict[sid][qid]
        """
        d = 100
        Pre_true={}
        Pre_false={}
        for qid, pred in pred_all.items():
            Pre_true[qid] = pred
            Pre_false[qid] = 1 - pred
        w_ij_matrix={}
        for i ,_ in pred_all.items():
            w_ij_matrix[i] = {}
            for j,_ in pred_all.items(): 
                w_ij_matrix[i][j] = 0
        for i,_ in pred_all.items():
            for j,_ in pred_all.items():
                criterion_true_1 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_true, 1)
                criterion_false_1 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_false, 1)
                criterion_true_0 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_true, 0)
                criterion_false_0 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_false, 0)
                tensor_11=torch.tensor(Pre_true[i],requires_grad=True)
                tensor_12=torch.tensor(Pre_true[j],requires_grad=True)
                loss_true_1 = criterion_true_1(tensor_11, torch.tensor(1.0))
                loss_false_1 = criterion_false_1(tensor_11, torch.tensor(0.0))
                loss_true_0 = criterion_true_0(tensor_12, torch.tensor(1.0))
                loss_false_0 = criterion_false_0(tensor_12, torch.tensor(0.0))
                loss_true_1.backward()
                grad_true_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_false_1.backward()
                grad_false_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_true_0.backward()
                grad_true_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                loss_false_0.backward()
                grad_false_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                diff_norm_00 = math.fabs(grad_true_1 - grad_true_0)
                diff_norm_01 = math.fabs(grad_true_1 - grad_false_0)
                diff_norm_10 = math.fabs(grad_false_1 - grad_true_0)
                diff_norm_11 = math.fabs(grad_false_1 - grad_false_0)
                Expect = Pre_false[i]*Pre_false[j]*diff_norm_00 + Pre_false[i]*Pre_true[j]*diff_norm_01 +Pre_true[i]*Pre_false[j]*diff_norm_10 + Pre_true[i]*Pre_true[j]*diff_norm_11
                w_ij_matrix[i][j] = d - Expect
        return w_ij_matrix

    def F_s_func(self,S_set,w_ij_matrix):
        res = 0.0
        for w_i in w_ij_matrix:
            if(w_i not in S_set):
                mx = float('-inf')
                for j in S_set:
                    if w_ij_matrix[w_i][j] > mx:
                        mx = w_ij_matrix[w_i][j]
                res +=mx
                
        return res

    def delta_q_S_t(self, question_id, pred_all,S_set,sampled_elements):
        """ get BECAT Questions weights delta
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            v: float, Each weight information
        """     
        
        Sp_set = list(S_set)
        b_array = np.array(Sp_set)
        sampled_elements = np.concatenate((sampled_elements, b_array), axis=0)
        if question_id not in sampled_elements:
            sampled_elements = np.append(sampled_elements, question_id)
        sampled_dict = {key: value for key, value in pred_all.items() if key in sampled_elements}
        
        w_ij_matrix = self.get_BE_weights(sampled_dict)
        
        F_s = self.F_s_func(Sp_set,w_ij_matrix)
        
        Sp_set.append(question_id)
        F_sp =self.F_s_func(Sp_set,w_ij_matrix)
        return F_sp - F_s

    def bobcat_policy(self,S_set,untested_questions):
        """ get expected model change
        Args:
            S_set:list , the questions have been chosen
            untested_questions: dict, untested_questions
        Returns:
            float, expected model change
        """
        device = self.config['device']
        action_mask = [0.0] * self.n_q
        train_mask=[-0.0]*self.n_q
        for index in untested_questions:
            action_mask[index] = 1.0
        for state in S_set:
            keys = list(state.keys())
            key = keys[0]
            values = list(state.values())
            val = values[0]
            train_mask[key] = (float(val)-0.5)*2 
        action_mask = torch.tensor(action_mask).to(device)
        train_mask = torch.tensor(train_mask).to(device)
        _, action = self.policy.policy(train_mask, action_mask)
        return action.item()
    
    def cal_loss(self,student_id,exercise_id):
        id_stu_emb = torch.sigmoid(self.model.student_emb(student_id))
        id_k_difficulty = torch.sigmoid(self.model.k_difficulty(exercise_id))
        id_e_discrimination = torch.sigmoid(self.model.e_discrimination(exercise_id)) * 10
        stu_emb = torch.sigmoid(self.model.LLM_transfer_stu(self.model.init_stu[student_id]))
        k_difficulty = torch.sigmoid(self.model.LLM_transfer_k_difficulty(self.model.init_exer[exercise_id]))
        e_discrimination = torch.sigmoid(self.model.LLM_transfer_e_discrimination(self.model.init_exer[exercise_id])) * 10
        data={
            "stu":[stu_emb,id_stu_emb],
            "exer":[k_difficulty,id_k_difficulty],
            "disc":[e_discrimination,id_e_discrimination]
        }
        loss=0
        loss+=self.con_loss(data['exer'])
        return loss*self.alpha
    
    def con_loss(self,x_y, temp=1.0):
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
