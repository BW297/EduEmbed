import torch
from model.kancd import kancd_model
from model.orcdf import orcdf_model
from model.mirt import mirt_model
import wandb
import numpy as np
import argparse
from pprint import pprint
from data_set import DATA_SET
import pandas as pd
import json
import os

os.environ['WANDB_MODE']='offline'
parser = argparse.ArgumentParser()
parser.add_argument('--method',type=str, help='prediction method', required=True)
parser.add_argument('--train_file',type=str, help='train file list', required=True)
parser.add_argument('--test_file',type=str, help='test file list', required=True)
parser.add_argument('--epoch_num', type=int, help='epoch of method', default=20, required=True)
parser.add_argument('--seed', default=0, type=int, help='seed for exp', required=False)
parser.add_argument('--dtype', default=torch.float64, help='dtype of tensor', required=False)
parser.add_argument('--device', default='cuda', type=str, help='device for exp', required=True)
parser.add_argument('--latent_dim', type=int, help='dimension of hidden layer', default=64, required=True)
parser.add_argument('--batch_size', type=int, help='batch size of benchmark', default=256, required=True)
parser.add_argument('--lr', type=float, help='learning rate', default=1e-3, required=True)
parser.add_argument('--dropout', type=float, help='dropout', default=0.5, required=False)
parser.add_argument('--inter', type=str, help='orcdf interfunction', default='kancd', required=True)

parser.add_argument('--lamda', type=float, default=1, required=True)
parser.add_argument('--alpha', type=float, default=0.01, required=True)

parser.add_argument('--gcn_layers', type=int, help='numbers of gcn layers', default=3, required=False)
parser.add_argument('--keep_prob', type=float, default=1.0, help='edge drop probability', required=False)
parser.add_argument('--noise_ratio', type=float, default=0, help='the proportion of noise which added into response logs', required=False)
parser.add_argument('--weight_decay', type=float, default=0, required=False)
parser.add_argument('--ssl_temp', type=float, default=0.5, required=False)
parser.add_argument('--ssl_weight', type=float, default=1e-3, required=False)
parser.add_argument('--flip_ratio', type=float, default=0.15, required=False)
parser.add_argument('--orcdf_mode', type=str, default='', required=False)
parser.add_argument('--model_type', type=str, default='', required=False)

config_dict = vars(parser.parse_args())
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

method = config_dict["method"]
seed = config_dict["seed"]
batch_size = config_dict["batch_size"]
device = config_dict["device"]
dtype = config_dict["dtype"]
latent_dim = config_dict["latent_dim"]
epoch_num = config_dict["epoch_num"]
lr = config_dict["lr"]
train_file = config_dict["train_file"]
test_file = config_dict["test_file"]
model_type =config_dict['model_type']
lamda=config_dict['lamda']
alpha=config_dict['alpha']

train_file_list = train_file.split(',')
test_file_list = test_file.split(',')
file_list = train_file_list + test_file_list

set_seed(seed)

wandb.init(project="TransductiveCD", name=f"{train_file}--{test_file}--{method}--{batch_size}--{lamda}--{alpha}--{lr}--{seed}", config=config_dict)
pprint(config_dict)
train_file_list = train_file.split(',')
test_file_list = test_file.split(',')
file_list = train_file_list + test_file_list
print(train_file_list, test_file_list)
print(file_list)

data_set = DATA_SET(train_file_list, test_file_list, model_type)
if method == 'orcdf':
    cd = orcdf_model(data_set, **config_dict)
    cd.train_model(batch_size, epoch_num, lr, device)
elif method == 'kancd':
    cd = kancd_model(data_set, **config_dict)
    cd.train_model(batch_size, epoch_num, lr, device)
elif method == 'mirt':
    cd = mirt_model(data_set, **config_dict)
    cd.train_model(batch_size, epoch_num, lr, device)   

