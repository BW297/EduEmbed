import torch
from model.ncd import ncd_model
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
parser.add_argument('--method', default='ncd', type=str, help='prediction method', required=True)
parser.add_argument('--train_file', type=str, help='train file list', required=True)
parser.add_argument('--test_file',type=str, help='test file list', required=True)
parser.add_argument('--epoch_num', type=int, help='epoch of method', default=20, required=True)
parser.add_argument('--seed', default=0, type=int, help='seed for exp', required=False)
parser.add_argument('--dtype', default=torch.float64, help='dtype of tensor', required=False)
parser.add_argument('--device', default='cuda', type=str, help='device for exp', required=True)
parser.add_argument('--batch_size', type=int, help='batch size of benchmark', default=256, required=True)
parser.add_argument('--lr', type=float, help='learning rate', default=5e-4, required=True)
parser.add_argument('--inter', type=str, help='interfunction', default='ncd', required=True)
parser.add_argument('--model_type', type=str, default='', required=True)
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
epoch_num = config_dict["epoch_num"]
lr = config_dict["lr"]
train_file = config_dict["train_file"]
test_file = config_dict["test_file"]
model_type = config_dict['model_type']
train_file_list = train_file.split(',')
test_file_list = test_file.split(',')
file_list = train_file_list + test_file_list

set_seed(seed)

wandb.init(project="InductiveCD", name=f"{train_file}--{test_file}--{method}--{batch_size}--{lr}--{seed}", config=config_dict)
pprint(config_dict)
train_file_list = train_file.split(',')
test_file_list = test_file.split(',')
file_list = train_file_list + test_file_list
print(train_file_list, test_file_list)
print(file_list)

data_set = DATA_SET(train_file_list, test_file_list, model_type)
if method == 'ncd':
    cd = ncd_model(data_set, **config_dict)
    cd.train_model(batch_size, epoch_num, lr, device)
else:
    print("invalid method")