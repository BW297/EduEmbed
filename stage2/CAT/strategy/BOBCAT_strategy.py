import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from strategy.abstract_strategy import AbstractStrategy
from model.abstract_model import AbstractModel
from dataset.adaptest_dataset import AdapTestDataset

class BOBCAT(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'BOBCAT'
    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset,S_set):
        selection = {}
        for sid in tqdm(adaptest_data.data.keys(), "Selecting: "):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            j = model.bobcat_policy(S_set[sid],untested_questions)
            selection[sid] = j
        return selection