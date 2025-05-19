#!/bin/bash

# DATA PATHS
TRAIN_CSV_PATH="datasets/r1.train.csv"
VALID_CSV_PATH="datasets/r1.valid.csv"
TEST_CSV_PATH="datasets/r1.test.csv"

# Hyperparameters
BATCH_SIZE=8
EPOCHS=10
LR=1e-5

python main.py \
    --train_csv_path $TRAIN_CSV_PATH \
    --val_csv_path $VALID_CSV_PATH \
    --test_csv_path $TEST_CSV_PATH \
    --audio_column "audio" \
    --label_column "new_label" \
    --text_column "target_text" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --model_path "0519_contrastive_wav2vec2.pt" \
    --wandb \
    --run_inference \
    --train \
    # --test_saved_checkpoint \

