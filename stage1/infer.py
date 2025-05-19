import torch
from transformers import AutoConfig
from model.llm_model import finetuneModel
import torch.nn as nn
import numpy as np
import os
import pickle
from safetensors.torch import load_file
from utils import infer_args
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
config=infer_args()
import torch
import torch.nn as nn

class FineTuned_Model_Infer(nn.Module):
    def __init__(self, **config_dict):
        super(FineTuned_Model_Infer, self).__init__()
        self.data_type = config_dict['data_file'].split(',')
        self.model_type = config_dict['model_type']
        self.base_model_path=config_dict['base_model_path']
        self.pooling_type = config_dict['pooling_type']
        self.use_Lora = config_dict['using_Lora']
        self.using_score=config_dict['using_score']
        self.data_train_type=config_dict['data_train_type']
        self.finetune_path=config_dict['finetune_path']
        config = AutoConfig.from_pretrained(self.base_model_path)
        if self.using_score:
            self.model =  finetuneModel(config_dict['base_model_path'], config=config, pooling_type=config_dict['pooling_type'], output_dim=64, use_Lora=config_dict['using_Lora'])
            self.model = self.model.to(device)  

        state_dict = load_file(self.finetune_path)
        self.model.load_state_dict(state_dict)

    def forward(self):
        self.model.eval()
        for idx,data_type in enumerate(self.data_type):
            with open(data_type+'/'+self.model_type+'/output.pkl', 'rb') as file:
                self.data_set = pickle.load(file)
            folder_path = data_type+'/'+self.data_train_type
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            with open(folder_path+"/stu_embedding.csv", mode='a+') as f:
                batch_embeddings = []
                batch_size = 16
                for batch_data in self.data_set['stu']:
                    model_list = []
                    input_ids_batch = batch_data['input_ids']
                    for i in range(0, input_ids_batch.shape[0], batch_size):
                        input_ids_batch = batch_data['input_ids'][i:i+batch_size].to(device)  
                        attention_mask = batch_data['attention_mask'][i:i+batch_size].to(device)
                        entity_type_ids = batch_data['entity_type_ids'][i:i+batch_size].to(device)
                        input_state = self.model(
                            input_ids=input_ids_batch,
                            attention_mask=attention_mask,
                            entity_type_ids=entity_type_ids
                        )
                        model_list.append(input_state.detach())

                    out_put = torch.cat(model_list, dim=0)
                    mean_embedding = torch.mean(out_put, dim=0)  
                    batch_embeddings.append(mean_embedding.cpu().numpy())  
                    

                np.savetxt(f, batch_embeddings, delimiter=",")

            with open(folder_path+"/exer_embedding.csv", mode='a+') as f:
                batch_embeddings = []
                batch_size = 16
                for batch_data in self.data_set['exer']:
                    model_list = []
                    input_ids_batch = batch_data['input_ids']

                    for i in range(0, input_ids_batch.shape[0], batch_size):
                        input_ids_batch = batch_data['input_ids'][i:i+batch_size].to(device)
                        attention_mask = batch_data['attention_mask'][i:i+batch_size].to(device)
                        entity_type_ids = batch_data['entity_type_ids'][i:i+batch_size].to(device)
                        input_state = self.model(
                            input_ids=input_ids_batch,
                            attention_mask=attention_mask,
                            entity_type_ids=entity_type_ids
                        )      
                        model_list.append(input_state.detach())

                    mean_embedding = torch.cat(model_list, dim=0)
                    np.savetxt(f, mean_embedding.cpu().numpy(), delimiter=",")

            with open(folder_path+"/know_embedding.csv", mode='a+') as f:
                batch_size = 16
                for batch_data in self.data_set['know']:
                    model_list = []
                    input_ids_batch = batch_data['input_ids']

                    for i in range(0, input_ids_batch.shape[0], batch_size):
                        input_ids_batch = batch_data['input_ids'][i:i+batch_size].to(device)
                        attention_mask = batch_data['attention_mask'][i:i+batch_size].to(device)
                        entity_type_ids = batch_data['entity_type_ids'][i:i+batch_size].to(device)
                        input_state = self.model(
                            input_ids=input_ids_batch,
                            attention_mask=attention_mask,
                            entity_type_ids=entity_type_ids
                        )      
                        model_list.append(input_state.detach())

                    mean_embedding = torch.cat(model_list, dim=0)
                    np.savetxt(f, mean_embedding.cpu().numpy(), delimiter=",")
llm_vec = FineTuned_Model_Infer(**config)
llm_vec.forward()
