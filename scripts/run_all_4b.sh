export https_proxy=http://10.217.142.137:8080
export WANDB_MODE=online

# Run all dynamic rollout experiments for Qwen3-4B in sequence
# Order: dynamic_lh(12/25/50) → dynamic(12/25/50) → high_entropy(12/25/50) → random(12/25/50)

run_experiment() {
    local MODE=$1
    local RATIO=$2
    echo "========================================================"
    echo "Starting: mode=${MODE}  ratio=${RATIO}  model=Qwen3-4B"
    echo "========================================================"

    accelerate launch \
        --config_file accelerate.yaml \
        --num_processes 8 \
        --gradient_accumulation_steps 1 \
        --main_process_port 12949 \
        opsd_train.py \
        --model_name_or_path /home/sankuai/buyixin02/egsd/model/Qwen3-4B \
        --learning_rate 5e-6 \
        --max_grad_norm 0.1 \
        --per_device_train_batch_size 4 \
        --gradient_checkpointing \
        --gradient_accumulation_steps 1 \
        --output_dir /home/sankuai/buyixin02/rollout_opsd/output/ \
        --run_config qwen34b_gen2048_fixteacher_temp11_forwardbeta0_clip005 \
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
        --rollout_keep_ratio ${RATIO} \
        --rollout_select_mode ${MODE} \
        --wandb_project OPSD

    echo "Finished: mode=${MODE}  ratio=${RATIO}"
    echo ""
}

# dynamic_lh: 12% → 25% → 50%
run_experiment dynamic_lh 0.12
run_experiment dynamic_lh 0.25
run_experiment dynamic_lh 0.5

# dynamic: 12% → 25% → 50%
run_experiment dynamic 0.12
run_experiment dynamic 0.25
run_experiment dynamic 0.5

# high_entropy: 12% → 25% → 50%
run_experiment high_entropy 0.12
run_experiment high_entropy 0.25
run_experiment high_entropy 0.5

# random: 12% → 25% → 50%
run_experiment random 0.12
run_experiment random 0.25
run_experiment random 0.5

echo "All experiments completed!"
