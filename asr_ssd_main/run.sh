#!/bin/bash

for fold in {0..1}
do 
    seed=42
    echo "running for fold $fold with seed $seed"

    CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=0 python main.py \
    --epochs 30 \
    --batch_size 8 \
    --target 'human_text_jamo' \
    --asr_mode 'human' \
    --train_filename "five_fold_datasets/test_fold_${fold}_train.csv" \
    --test_filename "five_fold_datasets/test_fold_${fold}.csv" \
    --num_runs 1 \
    --seed $seed 
done 

# --multitask_alpha 0.15 \
# --attention 'single' \
# --loss_feature 'binary_label' \
# --loss_feature 'age_label' \
# --embedding \
# --age_loss_weighting
# --reverse_alpha 0.5 \
# --train_filename 'tacotron_tts_0421_baseline_44_fn_sigma1.csv' \
