import numpy as np
import random
from tqdm import tqdm

from strategy.abstract_strategy import AbstractStrategy
from model.abstract_model import AbstractModel
from dataset.adaptest_dataset import AdapTestDataset


class MAATStrategy(AbstractStrategy):

    def __init__(self, n_candidates=10):
        super().__init__()
        self.n_candidates = n_candidates

    @property
    def name(self):
        return 'Model Agnostic Adaptive Testing'

    def _compute_coverage_gain(self, sid, qid, adaptest_data: AdapTestDataset):
        concept_cnt = {}
        for q in adaptest_data.data[sid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] = 0
        for q in list(adaptest_data.tested[sid]) + [qid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] += 1
        return (sum(cnt / (cnt + 1) for c, cnt in concept_cnt.items())
                / sum(1 for c in concept_cnt))

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        assert hasattr(model, 'expected_model_change'), \
            'the models must implement expected_model_change method'
        pred_all = model.get_pred(adaptest_data)
        selection = {}
        total_stu = adaptest_data.data.keys()
        # policy_stu = random.sample(total_stu, 10)
        # random_stu = [stu for stu in total_stu if stu not in policy_stu]
        for sid in tqdm(total_stu, "Policy Selecting: "):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            emc_arr = [model.expected_model_change(sid, qid, adaptest_data, pred_all) for qid in untested_questions]
            candidates = untested_questions[np.argsort(emc_arr)[::-1][:self.n_candidates]]
            selection[sid] = max(candidates, key=lambda qid: self._compute_coverage_gain(sid, qid, adaptest_data))
        # for sid in tqdm(random_stu, "Random Selecting: "):
        #     untested_questions = np.array(list(adaptest_data.untested[sid]))
        #     choice = np.random.randint(len(untested_questions))
        #     selection[sid] = untested_questions[choice]
        return selection