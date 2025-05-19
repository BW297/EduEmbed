import torch
import pandas as pd
from transformers import AutoTokenizer
import json
import numpy as np
import pickle
from torch.utils.data import TensorDataset
import os
import random
random.seed(42)
class Data_Process:
    def __init__(self, file, base_model_path, model_type, OOM):
        self.file=file.split(",")
        self.model_type = model_type
        self.OOM=OOM
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

    def load_sentences(self,file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file.readlines()]

    def classification(self):
        for file_name in self.file:
            stu_data = []
            exer_data = []
            know_data = []
            with open(file_name+"/student_exer.json", "r", encoding="utf-8") as file:
                stu_dict = json.load(file)
            stu_embeddings = self.load_sentences(file_name+'/student.txt')
            stu_embeddings=np.array(stu_embeddings)
            for key, value in stu_dict.items():
                data = {}
                stu_idx = np.array(value)
                stu_re = stu_embeddings[stu_idx].tolist()
                if self.OOM:
                    stu_sample = stu_re if len(stu_re) < 50 else random.sample(stu_re, 50)
                    stu_re = stu_sample
                if len(stu_idx)!=0:
                    stu_inputs = self.tokenizer(stu_re, return_tensors="pt", padding='max_length', truncation=True, max_length=64)
                    full_of_2 = torch.full_like(stu_inputs['input_ids'], 2)
                else:
                    stu_re=['This student has no relevant response records.']
                    stu_inputs = self.tokenizer(stu_re, return_tensors="pt", padding='max_length', truncation=True, max_length=64)
                    full_of_2 = torch.full_like(stu_inputs['input_ids'], 2)
                data['entity_type_ids'] = full_of_2
                data['attention_mask'] = stu_inputs['attention_mask']
                data['input_ids'] = stu_inputs['input_ids']
                data['num'] = len(stu_idx)
                if self.OOM:
                    data['num'] = len(stu_idx) if len(stu_idx) <50 else 50
                data['label'] = 2
                stu_data.append(data)

            exer_embeddings = self.load_sentences(file_name + '/exercise.txt')
            for sentence in exer_embeddings:
                sentence=[sentence]
                exer_inputs = self.tokenizer(sentence, return_tensors="pt", padding='max_length', truncation=True, max_length=64)
                full_of_1 = torch.full_like(exer_inputs['input_ids'], 1)
                exer_data.append({
                    "input_ids": exer_inputs['input_ids'],
                    "attention_mask": exer_inputs['attention_mask'],
                    "entity_type_ids": full_of_1,
                    "num": 1,
                    "label": 1
                })

            know_embeddings = self.load_sentences(file_name + '/concept.txt')
            know_inputs = self.tokenizer(know_embeddings, return_tensors="pt", padding='max_length', truncation=True, max_length=64)
            full_of_0 = torch.full_like(know_inputs['input_ids'], 0)
            know_data.append({
                "input_ids": know_inputs['input_ids'],
                "attention_mask": know_inputs['attention_mask'],
                "entity_type_ids": full_of_0,
            })

            stu_count, exer_count, know_count = len(stu_data), len(exer_data), len(know_data)
            print(f"stu_count: {stu_count}, exer_count: {exer_count}, know_count: {know_count}")
            data_set={}

            data_set['stu']=stu_data
            data_set['exer']=exer_data
            data_set['know']=know_data
            folder_path = file_name+'/'+self.model_type
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            with open(folder_path+'/output.pkl', 'wb') as file:
                pickle.dump(data_set, file)

    def get_q_matrix(self):
        exer_n, know_n = 0, 0
        for file in self.file:
            with open(file + "/data.json", 'r') as json_file:
                df = json.load(json_file)
                exer_n += df["exercise_num"]
                know_n += df["knowledge_num"]
        q_matrix = torch.zeros(size=(exer_n, know_n),dtype=torch.float64)
        exer_n, know_n = 0, 0
        for file in self.file:
            qf = pd.read_csv(file + "/q_matrix.csv", header=None)
            q_matrix = torch.tensor(qf.values)
            exer, know = q_matrix.shape
            q_matrix[exer_n:exer_n + exer, know_n:know_n + know] = q_matrix
            with open(file + "/data.json", 'r') as json_file:
                df = json.load(json_file)
                exer_n += df["exercise_num"]
                know_n += df["knowledge_num"]
        return q_matrix

    def get_response(self):
        train_response, test_response=self.get_data()
        q_matrix = self.get_q_matrix()
        user=train_response.iloc[:,0]
        item=train_response.iloc[:,1]
        score = train_response.iloc[:,2]
        train_dataset = TensorDataset(
            torch.tensor(user.apply(int), dtype=torch.int64),
            torch.tensor(item.apply(int), dtype=torch.int64),
            q_matrix[np.array(item, dtype=int), :],
            torch.tensor(score.apply(int), dtype=torch.float64),
        )

        user=test_response.iloc[:,0]
        item=test_response.iloc[:,1]
        score = test_response.iloc[:,2]
        test_dataset = TensorDataset(
            torch.tensor(user.apply(int), dtype=torch.int64),
            torch.tensor(item.apply(int), dtype=torch.int64),
            q_matrix[np.array(item, dtype=int), :],
            torch.tensor(score.apply(int), dtype=torch.float64),
        )

        
        return train_dataset, test_dataset
    
    def get_data(self):
        dfs = []
        add_stu_num, add_exer_num, add_know_num = 0, 0, 0
        for index, file in enumerate(self.file):
            df = pd.read_csv(file + "/train_response.csv", header=None)
            df.iloc[:, 0] += add_stu_num
            df.iloc[:, 1] += add_exer_num
            df.iloc[:, 2] += add_know_num
            with open(file + "/data.json", 'r') as json_file:
                json_f = json.load(json_file)
            add_stu_num += json_f["student_num"]
            add_exer_num += json_f["exercise_num"]
            add_know_num += json_f['knowledge_num']
            dfs.append(df)
        train_data = pd.concat(dfs, ignore_index=True)

        dfs = []
        add_stu_num, add_exer_num, add_know_num = 0, 0, 0
        for index, file in enumerate(self.file):
            df = pd.read_csv(file + "/test_response.csv", header=None)
            df.iloc[:, 0] += add_stu_num
            df.iloc[:, 1] += add_exer_num
            df.iloc[:, 2] += add_know_num
            with open(file + "/data.json", 'r') as json_file:
                json_f = json.load(json_file)
            add_stu_num += json_f["student_num"]
            add_exer_num += json_f["exercise_num"]
            add_know_num += json_f['knowledge_num']
            dfs.append(df)
        test_data = pd.concat(dfs, ignore_index=True)
        return train_data, test_data