from pprint import pprint
from data_process.data_pro import Data_Process
from utils import  data_args


def train(config):
    pprint(config)
    data_process = Data_Process(file=config['data_file'], base_model_path=config['base_model_path'], model_type=config['model_type'],OOM=config['OOM'])
    data_process.classification()
    print("Data Load!")
   

if __name__ == "__main__":
    config_dict = data_args()
    train(config_dict)