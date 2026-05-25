export https_proxy=http://10.217.142.137:8080
export WANDB_MODE=online

# Both student and teacher use non-thinking mode (enable_thinking=False)
accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 2 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path /home/sankuai/buyixin02/egsd/model/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir  /home/sankuai/buyixin02/rollout_opsd/output/ \
    --run_config qwen31b_gen2048_fixteacher_temp11_forwardbeta0_clip005_nonthink \
    --max_steps 100 \
    --max_completion_length 2048 \
    --save_steps 5 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --student_thinking False \
    --teacher_thinking False \
    --wandb_project OPSD
