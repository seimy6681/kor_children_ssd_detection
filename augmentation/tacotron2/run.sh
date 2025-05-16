CUDA_VISIBLE_DEVICES=1 python train.py \
    --output_directory='/data/selinawisco/25_tacotron2/0419_run' \
    --log_directory='logs' \
    --n_gpus=1 \
    --epochs=500 \
    --checkpoint_path '/data/selinawisco/25_tacotron2/0419_run/checkpoint_31000' \
    # --training_files='/data/selinawisco/kochild/data_train.txt' \ 
    # --validation_files='/data/selinawisco/kochild/data_test.txt'\
    # -- 
    