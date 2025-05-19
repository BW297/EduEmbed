deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 8333 \
    main.py \
    --deepspeed deep_speed.json \
    --model_type $Abbreviation_of_LM$ \
    --base_model_path $Path_to_base_LM$ \
    --data_file $Path_to_dataset$ \
    --pooling_type causal \
    --use_grad_accum \
    --using_score \
    --using_Lora \
    --OOM
