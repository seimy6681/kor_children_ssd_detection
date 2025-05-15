CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=0 python main.py \
--epochs 30 \
--batch_size 8 \
--target 'human_text_jamo' \
--asr_mode 'human' \
--train_filename 'r08_APAC_KAPP_25_train.csv' \
--test_filename 'r08_APAC_KAPP_25_test.csv' \
--num_runs 1 \
--seed 44 \
# --multitask_alpha 0.3 \
# --loss_feature 'age_label' \
# --embedding \
# --age_loss_weighting
# --reverse_alpha 0.5 \
# --train_filename 'tacotron_tts_0421_baseline_44_fn_sigma1.csv' \
# --attention 'single' \
# --loss_feature 'binary_label' \
