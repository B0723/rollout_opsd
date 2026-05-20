export https_proxy=http://10.217.142.137:8080
export WANDB_MODE=online

# Best-of-N OPSD — Qwen3-1.7B  4 GPU  effective_bs=32
# Generates bon_n rollouts per problem, selects best by (1-L_hat)*(1-H_hat),
# Teacher forward runs only on the selected rollout per problem.
# Usage:
#   bash run_bon_1b.sh        # default N=4
#   bash run_bon_1b.sh 8      # N=8
BON_N=${1:-4}

# NOTE: Student forward processes per_device_batch*bon_n sequences per GPU.
# With per_device_batch=4 and bon_n=4: 16 sequences → ~9.3 GB student log_probs.
# Reduce per_device_train_batch_size if OOM (e.g. to 2 with ga=4 to keep effective_bs=32).

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
    --output_dir /home/sankuai/buyixin02/rollout_opsd/output/ \
    --run_config qwen31b_gen2048_fixteacher_temp11_forwardbeta0_clip005 \
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
    --bon_n ${BON_N} \
    --wandb_project OPSD
