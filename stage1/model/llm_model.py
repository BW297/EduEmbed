from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.scoring_layer = nn.Sequential(
            nn.Linear(input_dim, 2048, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(2048, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(1024, output_dim, bias=False)
        )
        
    def forward(self, x):
        scores = self.scoring_layer(x)
        return scores
    
class finetuneModel(torch.nn.Module):
    def __init__(self, base_model_path, config, pooling_type, output_dim, use_Lora):
        super(finetuneModel, self).__init__()
        model = AutoModel.from_pretrained(base_model_path)
        self.pooling_type = pooling_type
        self.classifier = Classifier(config.hidden_size, output_dim)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            model.resize_token_embeddings(len(tokenizer))
        if use_Lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.base_model = get_peft_model(model, lora_config)
        else:
            self.base_model=model

        self.entity_type_embedding = torch.nn.Embedding(
            num_embeddings=3,
            embedding_dim=config.hidden_size
        ,device=self.base_model.device)
        torch.nn.init.xavier_normal_(self.entity_type_embedding.weight)

    def forward(self, input_ids, attention_mask, entity_type_ids):
        base_embeddings = self.base_model.get_input_embeddings()(input_ids)
        
        new_embeddings = self.entity_type_embedding(entity_type_ids)

        combined_embeddings = base_embeddings + new_embeddings

        outputs = self.base_model(
            attention_mask=attention_mask,
            inputs_embeds=combined_embeddings
        ).last_hidden_state
        
        if self.pooling_type=='cls':
            outputs=outputs[:,0,:]
        elif self.pooling_type=='average':
            attention=attention_mask
            attention = attention.unsqueeze(-1).expand(outputs.size())
            masked_output = outputs * attention
            sum_embeddings = masked_output.sum(dim=1)
            valid_token_counts = attention.sum(dim=1)
            outputs = sum_embeddings / valid_token_counts
        elif self.pooling_type=='causal':
            last_valid_index = attention_mask.sum(dim=1) - 1
            outputs = outputs[torch.arange(outputs.size(0)), last_valid_index, :]
        outputs=self.classifier(outputs)
        return outputs
