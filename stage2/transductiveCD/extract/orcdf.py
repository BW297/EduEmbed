import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

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
class ORCDF_Extractor(nn.Module):
    def __init__(self, dataset, **dict):
        super().__init__()
        self.dataset=dataset
        self.latent_dim = dict['latent_dim']
        self.lam=dict['lamda']
        self.alpha=dict['alpha']
        self.ssl_temp = dict['ssl_temp']
        self.ssl_weight = dict['ssl_weight']
        self.mode = dict['orcdf_mode']
        self.emb_map = {}
        self.device = dict['device']
        self.dtype = dict['dtype']
        self.gcn_layers = dict['gcn_layers']
        self.keep_prob = dict['keep_prob']
        self.gcn_drop = True
        self.graph_dict = ...
        self.tr_stu_n, self.tr_exer_n, self.te_stu_n, self.te_exer_n, self.tr_known, self.known = dataset.get_num()

        train_emb, test_emb = dataset.get_text_embedding(self.dtype, self.device)
        self.dict = {
            "train": train_emb,
            "test": test_emb
        }
        self.text_dim = self.dict["train"]["student"].shape[1]
        self.__student_emb = get_mlp(self.text_dim, self.latent_dim, self.dtype, self.device)
        self.__knowledge_emb = get_mlp(self.text_dim, self.latent_dim, self.dtype, self.device)
        self.__exercise_emb = get_mlp(self.text_dim, self.latent_dim, self.dtype, self.device)
        self.__disc_emb = get_mlp(self.text_dim, 1, self.dtype, self.device)

        self.know_emb = nn.Parameter(torch.zeros(self.tr_known, self.latent_dim,dtype=self.dtype)).to(self.device)
        self.exer_emb=nn.Embedding(self.tr_exer_n,self.latent_dim,dtype=self.dtype).to(self.device)
        self.stu_emb=nn.Embedding(self.tr_stu_n,self.latent_dim,dtype=self.dtype).to(self.device)
        self.disc_emb =nn.Embedding(self.tr_exer_n,1,dtype=self.dtype).to(self.device)
        
        self.know_concat=nn.Linear(2*self.latent_dim,self.latent_dim,dtype=self.dtype).to(self.device)
        self.exer_concat=nn.Linear(2*self.latent_dim,self.latent_dim,dtype=self.dtype).to(self.device)
        self.stu_concat=nn.Linear(2*self.latent_dim,self.latent_dim,dtype=self.dtype).to(self.device)
        self.disc_concat =nn.Linear(2,1,dtype=self.dtype).to(self.device)
        
        self.concat_layer = nn.Linear(2 * self.latent_dim, self.latent_dim, dtype=self.dtype).to(self.device)

        self.apply(self.initialize_weights)

    def get_graph_dict(self, graph_dict):
        self.graph_dict = graph_dict

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def get_all_emb1(self):
        stu_emb, exer_emb, know_emb = (self.__student_emb(self.emb["student"]),
                                       self.__exercise_emb(self.emb["exercise"]),
                                       self.__knowledge_emb(self.emb["knowledge"]))
        
        all_emb = torch.cat([stu_emb, exer_emb, know_emb]).to(self.device)
        return all_emb
    
    def get_all_emb2(self):
        stu_emb, exer_emb, know_emb = (self.stu_emb.weight,
                                       self.exer_emb.weight,
                                       self.know_emb)
        
        all_emb = torch.cat([stu_emb, exer_emb, know_emb]).to(self.device)
        return all_emb

    def convolution(self, graph):
        all_emb = self.get_all_emb()
        emb = [all_emb]
        for layer in range(self.gcn_layers):
            all_emb = torch.sparse.mm(self.__graph_drop(graph), all_emb)
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return out_emb

    def __common_forward1(self, right, wrong):
        all_emb = self.get_all_emb1()
        emb = [all_emb]
        right_emb = all_emb
        wrong_emb = all_emb
        for layer in range(self.gcn_layers):
            right_emb = torch.sparse.mm(self.__graph_drop(right), right_emb)
            wrong_emb = torch.sparse.mm(self.__graph_drop(wrong), wrong_emb)
            all_emb = self.concat_layer(torch.cat([right_emb, wrong_emb], dim=1))
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return out_emb[:self.student_num], out_emb[self.student_num:self.student_num + self.exercise_num], out_emb[
                                                                                                           self.exercise_num + self.student_num:]
    def __common_forward2(self, right, wrong):
        all_emb = self.get_all_emb2()
        emb = [all_emb]
        right_emb = all_emb
        wrong_emb = all_emb
        for layer in range(self.gcn_layers):
            right_emb = torch.sparse.mm(self.__graph_drop(right), right_emb)
            wrong_emb = torch.sparse.mm(self.__graph_drop(wrong), wrong_emb)
            all_emb = self.concat_layer(torch.cat([right_emb, wrong_emb], dim=1))
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return out_emb[:self.student_num], out_emb[self.student_num:self.student_num + self.exercise_num], out_emb[
                                                                                                           self.exercise_num + self.student_num:]

    def __dropout(self, graph, keep_prob):
        if self.gcn_drop and self.training:
            size = graph.size()
            index = graph.indices().t()
            values = graph.values()
            random_index = torch.rand(len(values)) + keep_prob
            random_index = random_index.int().bool()
            index = index[random_index]
            values = values[random_index] / keep_prob
            g = torch.sparse.DoubleTensor(index.t(), values, size)
            return g
        else:
            return graph

    def __graph_drop(self, graph):
        g_dropped = self.__dropout(graph, self.keep_prob)
        return g_dropped

    def extract(self, student_id, exercise_id, type):
        if type == 'train':
            self.student_num, self.exercise_num = self.tr_stu_n, self.tr_exer_n
            self.emb = self.dict['train']
        else:
            self.emb = self.dict['test']
            self.student_num, self.exercise_num = self.te_stu_n, self.te_exer_n
            
        if 'dis' not in self.mode:
            stu_forward, exer_forward, know_forward = self.__common_forward1(self.graph_dict['right'],
                                                                            self.graph_dict['wrong'])
            stu_forward_flip, exer_forward_flip, know_forward_flip = self.__common_forward1(
                self.graph_dict['right_flip'],
                self.graph_dict['wrong_flip'])
            stu_forward1, exer_forward1, know_forward1 = self.__common_forward2(self.graph_dict['right'],
                                                                            self.graph_dict['wrong'])
            stu_forward_flip1, exer_forward_flip1, know_forward_flip1 = self.__common_forward2(
                self.graph_dict['right_flip'],
                self.graph_dict['wrong_flip'])
        else:
            out = self.convolution(self.graph_dict['all'])
            stu_forward, exer_forward, know_forward = out[:self.student_num], out[
                                                                            self.student_num:self.student_num + self.exercise_num], out[
                                                                                                                                    self.exercise_num + self.student_num:]


        extra_loss = 0

        def InfoNCE(view1, view2, temperature: float = 1.0, b_cos: bool = False):
            if b_cos:
                view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

            pos_score = (view1 @ view2.T) / temperature
            score = torch.diag(F.log_softmax(pos_score, dim=1))
            return -score.mean()

        if 'cl' not in self.mode:
            extra_loss1 = self.ssl_weight * (InfoNCE(stu_forward, stu_forward_flip, temperature=self.ssl_temp)
                                            + InfoNCE(exer_forward, exer_forward_flip,
                                                    temperature=self.ssl_temp))
            extra_loss2=0
            extra_loss2 = self.ssl_weight * (InfoNCE(stu_forward1, stu_forward_flip1, temperature=self.ssl_temp)
                                            + InfoNCE(exer_forward1, exer_forward_flip1,
                                                    temperature=self.ssl_temp))
            extra_loss=extra_loss1+extra_loss2

        student_ts1 = F.embedding(student_id, stu_forward)
        diff_ts1 = F.embedding(exercise_id, exer_forward)
        knowledge_ts1 = know_forward
        disc_ts1 = self.__disc_emb(self.emb['exercise'])[exercise_id]
        disc_ts2 = self.disc_emb(exercise_id)

        student_ts2 = F.embedding(student_id, stu_forward1)
        diff_ts2 = F.embedding(exercise_id, exer_forward1)
        knowledge_ts2 = know_forward1
       
        student_ts=self.lam*student_ts1+(1-self.lam)*student_ts2
        disc_ts=self.lam*disc_ts1+(1-self.lam)*disc_ts2
        diff_ts=self.lam*diff_ts1+(1-self.lam)*diff_ts2
        knowledge_ts=self.lam*(knowledge_ts1)+(1-self.lam)*knowledge_ts2
        return student_ts, diff_ts, disc_ts, knowledge_ts, {'extra_loss': extra_loss+self.cal_loss(student_id,exercise_id)}
    
    def update(self, item):
        if 'dis' not in self.mode:
            stu_forward, exer_forward, know_forward = self.__common_forward1(self.graph_dict['right'],
                                                                            self.graph_dict['wrong'])
            stu_forward1, exer_forward1, know_forward1= self.__common_forward2(self.graph_dict['right'],
                                                                            self.graph_dict['wrong'])
        else:
            out = self.convolution(self.graph_dict['all'])
            stu_forward, exer_forward, know_forward = (out[:self.student_num],
                                                       out[self.student_num:self.student_num + self.exercise_num],
                                                       out[self.exercise_num + self.student_num:])


        student_ts=self.lam*stu_forward+(1-self.lam)*stu_forward1
        knowledge_ts=self.lam*know_forward+(1-self.lam)*know_forward1

        self.emb_map["student"] = student_ts
        self.emb_map["knowledge"] = knowledge_ts
        return self.emb_map[item]
    
    def get_flip_graph(self, flag):
        def get_flip_data(data):
            import numpy as np
            np_response_flip = data.copy()
            column = np_response_flip[:, 2]
            probability = np.random.choice([True, False], size=column.shape,
                                           p=[self.graph_dict['flip_ratio'], 1 - self.graph_dict['flip_ratio']])
            #按照self.graph_dict['flip_ratio']的概率分配true或者false
            column[probability] = 1 - column[probability]#true则被反转
            np_response_flip[:, 2] = column
            return np_response_flip

        response_flip = get_flip_data(self.graph_dict['response'])
        se_graph_right_flip, se_graph_wrong_flip = [self.__create_adj_se(response_flip, flag,is_subgraph=True)[i] for i in
                                                    range(2)]
        ek_graph = self.graph_dict['Q_Matrix']
        self.graph_dict['right_flip'], self.graph_dict['wrong_flip'] = self.__final_graph(se_graph_right_flip,
                                                                                          ek_graph, flag), self.__final_graph(
            se_graph_wrong_flip, ek_graph, flag)

    @staticmethod
    def __get_csr(rows, cols, shape):
        values = np.ones_like(rows, dtype=np.float64)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)
    
    @staticmethod
    def __sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()
    
    def __create_adj_se(self, np_response, flag, is_subgraph=False):
        if flag == 'train':
            self.student_num, self.exercise_num, _, _, _, _ = self.dataset.get_num()
        else:
            _, _, self.student_num, self.exercise_num, _, _ = self.dataset.get_num()
        
        if is_subgraph:

            train_stu_right = np_response[np_response[:, 2] == 1, 0]
            train_exer_right = np_response[np_response[:, 2] == 1, 1]
            train_stu_wrong = np_response[np_response[:, 2] == 0, 0]
            train_exer_wrong = np_response[np_response[:, 2] == 0, 1]

            adj_se_right = self.__get_csr(train_stu_right, train_exer_right,
                                          shape=(self.student_num, self.exercise_num))
            adj_se_wrong = self.__get_csr(train_stu_wrong, train_exer_wrong,
                                          shape=(self.student_num, self.exercise_num))
            return adj_se_right.toarray(), adj_se_wrong.toarray()

        else:
            response_stu = np_response[:, 0]
            response_exer = np_response[:, 1]
            adj_se = self.__get_csr(response_stu, response_exer, shape=(self.student_num, self.exercise_num))
            return adj_se.toarray()

    def __final_graph(self, se, ek, flag):
        if flag == 'train':
            self.student_num, self.exercise_num, _, _, self.knowledge_num,_ = self.dataset.get_num()
        else:
            _, _, self.student_num, self.exercise_num,  self.knowledge_num,_ = self.dataset.get_num()
        
        sek_num = self.student_num + self.exercise_num + self.knowledge_num
        se_num = self.student_num + self.exercise_num
        tmp = np.zeros(shape=(sek_num, sek_num))
        tmp[:self.student_num, self.student_num:se_num] = se
        tmp[self.student_num:se_num, se_num:sek_num] = ek
        graph = tmp + tmp.T + np.identity(sek_num)
        graph = sp.csr_matrix(graph)

        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return self.__sp_mat_to_sp_tensor(adj_matrix).to(self.device)

    def get_id_and_get_text(self,student_id, exercise_id):
        stu_text=self.__student_emb(self.emb['student'])[student_id]
        stu_id=self.stu_emb(student_id)
        exer_text=self.__exercise_emb(self.emb['exercise'])[exercise_id]
        exer_id=self.exer_emb(exercise_id)
        know_text=self.__knowledge_emb(self.emb['knowledge'])
        know_id=self.know_emb
        disc_text=self.__disc_emb(self.emb['disc'])[exercise_id]
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
