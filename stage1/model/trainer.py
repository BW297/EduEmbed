from transformers import Trainer
import torch
import torch.nn as nn

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        def process_input(input_ids, attention_mask, entity_type_ids, batch_size=64):
            model_list=[]
            for i in range(0, input_ids.size(0), batch_size):
                input_state=model(
                    input_ids=input_ids[i:i+batch_size],
                    attention_mask=attention_mask[i:i+batch_size],
                    entity_type_ids=entity_type_ids[i:i+batch_size]
                )
                model_list.append(input_state)
            out_put=torch.cat(model_list, dim=0)
            return out_put
        stu_inputs = inputs['stu']['content']
        stu_output = process_input(
            stu_inputs['input_ids'], stu_inputs['attention_mask'], stu_inputs['entity_type_ids']
        )
        num = inputs['stu']['num'].tolist()
        stu_output = torch.stack([chunk.mean(dim=0) for chunk in torch.split(stu_output, num)])

        def process_simple_input(inputs_dict):
            input_state = model(**inputs_dict)
            return input_state

        exer_output = process_simple_input(inputs['exer']['content'])
        know_output  = process_simple_input(inputs['know'])
        stu_know_matrix = stu_output @ know_output.T  
        exer_know_matrix = exer_output @ know_output.T 
        performance = torch.sum((stu_know_matrix-exer_know_matrix)*inputs['q_matrix'],dim=1)
        performance=torch.sigmoid(performance)

        train_loss=nn.BCELoss()
        performloss = train_loss(performance , inputs['score'])
        return (performloss, None) if return_outputs else performloss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            def process_input(input_ids, attention_mask, entity_type_ids, batch_size=64):
                model_list=[]
                for i in range(0, input_ids.size(0), batch_size):
                    input_state=model(
                        input_ids=input_ids[i:i+batch_size],
                        attention_mask=attention_mask[i:i+batch_size],
                        entity_type_ids=entity_type_ids[i:i+batch_size]
                    ).detach()
                    
                    model_list.append(input_state)
                out_put=torch.cat(model_list, dim=0)
                return out_put
            stu_inputs = inputs['stu']['content']
            stu_output = process_input(
                stu_inputs['input_ids'], stu_inputs['attention_mask'], stu_inputs['entity_type_ids']
            )
            num = inputs['stu']['num'].tolist()
            stu_output = torch.stack([chunk.mean(dim=0) for chunk in torch.split(stu_output, num)])
            def process_simple_input(inputs_dict):
                input_state = model(**inputs_dict).detach()
                return input_state
            exer_output = process_simple_input(inputs['exer']['content'])
            know_output  = process_simple_input(inputs['know'])
            stu_know_matrix = stu_output @ know_output.T  
            exer_know_matrix = exer_output @ know_output.T 
            performance = torch.sum((stu_know_matrix - exer_know_matrix) * inputs['q_matrix'], dim=1)
            performance=torch.sigmoid(performance)
            
        return None, performance.detach(),inputs['score']