import os
from pprint import pprint
from model.trainer import CustomTrainer
from data_process.data_collator import CustomDataCollator
from data_process.data_pro import Data_Process
from model.llm_model import finetuneModel
from utils import compute_metrics, get_args, set_seed, parse_args
from transformers import AutoConfig

os.environ["NCCL_P2P_LEVEL"] = "NVL"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["DS_SKIP_CUDA_CHECK"] = "1"

def train(config):

    pprint(config)

    set_seed(config['seed'])
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    data_process = Data_Process(file=config['data_file'], base_model_path=config['base_model_path'], model_type=config['model_type'],OOM=config['OOM'])
    train_data, test_data = data_process.get_response()
    model_config = AutoConfig.from_pretrained(config['base_model_path'])
    model = finetuneModel(config['base_model_path'], config=model_config, pooling_type=config['pooling_type'], output_dim=64, use_Lora=config['using_Lora'])
    training_args = get_args(config)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
        data_collator=CustomDataCollator(config['data_file'], model_type=config['model_type'])
    )
    trainer.train()

if __name__ == "__main__":
    config_dict = parse_args()
    train(config_dict)