python infer.py \
    --model_type $Abbreviation_of_LM$ \
    --base_model_path $Path_to_base_LM$ \
    --finetune_path $Path_to_finetuned_LM$ \
    --data_file $Path_to_dataset$ \
    --pooling_type causal \
    --using_Lora \
    --using_score