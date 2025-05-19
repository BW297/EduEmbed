import torch
from utils import Pro_Data
class CustomDataCollator(object):
    def __init__(self, file, model_type):
        file=file.split(",")
        self.data_set = Pro_Data(file, model_type)

    def __call__(self, instances):
        categories = ['stu', 'exer']

        batch_data = {
            cat: {
                "content": {
                    'input_ids': [],
                    'attention_mask': [],
                    'entity_type_ids': [],
                },
                'num': [],
                'label': [],
                'idx': []
            } for cat in categories
        }
        batch_data['q_matrix']=[]
        batch_data['score']=[]
        batch_data['know']={}

        for stu_idx, exer_idx, q_idx, score in instances:
            indices = {'stu': stu_idx, 'exer': exer_idx}
            for cat in categories:
                data = self.data_set[cat][indices[cat]]
                batch_data[cat]['content']['input_ids'].append(data['input_ids'])
                batch_data[cat]['content']['attention_mask'].append(data['attention_mask'])
                batch_data[cat]['content']['entity_type_ids'].append(data['entity_type_ids'])
                batch_data[cat]['num'].append(data['num'])
                batch_data[cat]['label'].append(data['label'])
                batch_data[cat]['idx'].append(indices[cat])
            batch_data['score'].append(score)
            batch_data['q_matrix'].append(q_idx)

        for cat in categories:
            batch_data[cat]['content']['input_ids'] = torch.cat(batch_data[cat]['content']['input_ids'], dim=0)
            batch_data[cat]['content']['attention_mask'] = torch.cat(batch_data[cat]['content']['attention_mask'], dim=0)
            batch_data[cat]['content']['entity_type_ids'] = torch.cat(batch_data[cat]['content']['entity_type_ids'], dim=0)
            batch_data[cat]['num'] = torch.tensor(batch_data[cat]['num'])
            batch_data[cat]['label'] = torch.tensor(batch_data[cat]['label'])
            batch_data[cat]['idx'] = torch.tensor(batch_data[cat]['idx'])


        batch_data['q_matrix'] = torch.stack(batch_data['q_matrix'], dim=0)
        batch_data['score'] = torch.tensor(batch_data['score'])
        batch_data['know']=self.data_set['know'][0]

        return batch_data
