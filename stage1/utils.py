import torch
import torch.nn as nn
from transformers import EvalPrediction, TrainingArguments
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error
import numpy as np
import pickle
import argparse

def compute_metrics(eval_pred: EvalPrediction, compute_result: bool = False):

    predictions, label = eval_pred.predictions, eval_pred.label_ids
    predictions = torch.tensor(predictions)
    label = torch.tensor(label)
    train_loss=nn.BCELoss()
    eval_loss = train_loss(predictions , label)
    eval_AUC = roc_auc_score(label.tolist(), predictions.tolist())
    eval_ACC=accuracy_score(label.tolist(), np.array(predictions.tolist())>=0.5)
    eval_F1_score = f1_score(label.tolist(), np.array(predictions.tolist())>=0.5)
    eval_MSE = mean_squared_error(label.tolist(), predictions.tolist())
    eval_RMSE = np.sqrt(eval_MSE) 
    return {"ACC": eval_ACC, 'AUC': eval_AUC, 'F1_score': eval_F1_score, 'loss': eval_loss, 'RMSE': eval_RMSE}

def get_args(config_dict):
    training_args = TrainingArguments(
        learning_rate=5e-5,
        run_name=config_dict['output_dir'],
        output_dir=config_dict['output_dir'],        
        num_train_epochs=config_dict['epoch_num'],         
        per_device_train_batch_size=config_dict['batch_size'],  
        per_device_eval_batch_size=config_dict['batch_size'],   
        weight_decay=0.05,             
        logging_steps=config_dict['logging_steps'],           
        eval_strategy="steps",         
        eval_steps=config_dict['step'],              
        save_steps=config_dict['step'],
        remove_unused_columns=False,
        fp16=True,                    
        metric_for_best_model="ACC",
        greater_is_better=True,
        gradient_accumulation_steps=config_dict['gradient_accumulation_steps'],
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        report_to="wandb",
        save_only_model=True,
        lr_scheduler_type="constant_with_warmup",
        ddp_broadcast_buffers=False
    )
    return training_args

def Pro_Data(file,base_model_path):
    stu_data=[]
    exer_data=[]
    know_data=[]
    data_set={}
    for file_name in file:
        with open(file_name+'/'+base_model_path+'/output.pkl', 'rb') as file:
            data_set = pickle.load(file)
            stu_data.extend(data_set['stu'])
            exer_data.extend(data_set['exer'])
            know_data.extend(data_set['know'])
    data_set['stu']=stu_data
    data_set['exer']=exer_data
    data_set['know']=know_data
    return data_set
    

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
def parse_args():
    parser = argparse.ArgumentParser(description="Train finetuneModel")
    parser.add_argument('--model_type', type=str, help='Abbreviation of fine-tuned model')
    parser.add_argument('--base_model_path', type=str, help='Path to the base model')
    parser.add_argument('--data_file', type=str, help='Dataset name')
    parser.add_argument('--epoch_num', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--step', type=int, default=500, help='Step number for evaluation and saving')
    parser.add_argument('--use_grad_accum', action='store_true', help="Enable gradient accumulation")
    parser.add_argument('--local_rank', type=int, help='Local process ID for distributed training')
    parser.add_argument('--deepspeed', type=str, help='Path to DeepSpeed config file')
    parser.add_argument('--pooling_type', type=str, help='Pooling type')
    parser.add_argument('--using_Lora', action='store_true', help='Use LoRA for parameter-efficient tuning')
    parser.add_argument('--using_score', action='store_true', help='Use classification head')
    parser.add_argument('--logging_steps', type=int, default=100, help="Logging interval (in steps)")
    parser.add_argument('--OOM', action='store_true', help='Flag for Out-of-Memory handling')
    args = vars(parser.parse_args())
    flags = []

    if args['using_Lora']:
        flags.append('Lora')
    if args['using_score']:
        flags.append('score')
    if args['use_grad_accum']:
        flags.append('grad_accum')

    flags.append(f"bs_{args['batch_size']}")
    flags.append(f"epoch_{args['epoch_num']}")
    flags.append(f"step_{args['step']}")
    flags.append(f"pool_{args['pooling_type']}")
    flags.append(f"log_{args['logging_steps']}")

    flags_str = '_'.join(flags)
    args['output_dir'] = f"checkpoint/{args['data_file'].replace('/', '_')}/{args['model_type']}_{flags_str}_results"
    args['model_type'] = f"{args['model_type']}"
    args['gradient_accumulation_steps'] = 16 if args['use_grad_accum'] else 1

    return args


def data_args():
    parser = argparse.ArgumentParser(description="General finetuneModel")
    parser.add_argument('--model_type', type=str, help='Abbreviation of fine-tuned model')
    parser.add_argument('--base_model_path', type=str, help='Path to the base model')
    parser.add_argument('--data_file', type=str, help='Dataset name')
    parser.add_argument('--OOM', action='store_true', help='Flag for Out-of-Memory handling')
    args = vars(parser.parse_args())
    return args


def infer_args():
    parser = argparse.ArgumentParser(description="Infer finetuneModel")
    parser.add_argument('--model_type', type=str, help='Abbreviation of fine-tuned model')
    parser.add_argument('--finetune_path', type=str, help='Path to fine-tuned model')
    parser.add_argument('--base_model_path', type=str, help='Path to the base model')
    parser.add_argument('--data_file', type=str, help='Dataset name')
    parser.add_argument('--pooling_type', type=str, help='Pooling type')
    parser.add_argument('--using_Lora', action='store_true', help='Use LoRA for parameter-efficient tuning')
    parser.add_argument('--using_score', action='store_true', help='Use classification head')
    args = vars(parser.parse_args())
    args['data_train_type'] = args['finetune_path'].split("/")[-3]
    return args
