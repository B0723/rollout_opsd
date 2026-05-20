#!/bin/bash

BASE_MODEL="/home/sankuai/buyixin02/egsd/model/Qwen3-1.7B"
CHECKPOINT_BASE="/home/sankuai/buyixin02/rollout_opsd/output/qwen31b_gen2048_fixteacher_temp11_forwardbeta0_clip005_dynamic50pct_dynamic50pct"

for CKPT in checkpoint-70 checkpoint-75 checkpoint-80 checkpoint-85 checkpoint-90 checkpoint-95 checkpoint-100; do
    echo "========== Evaluating $CKPT =========="
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --dataset "aime24" \
        --val_n 12 \
        --temperature 1.0 \
        --tensor_parallel_size 4 \
        --checkpoint_dir "$CHECKPOINT_BASE/$CKPT"
    wait
done
