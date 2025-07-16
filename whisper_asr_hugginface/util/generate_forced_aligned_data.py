import os
import pandas as pd
from forced_aligner import get_forced_alignment_data
from tqdm import tqdm

for fold in tqdm(range(1,5)):
# 자모로 나눌 word-level CSV 경로
    word_input_dir = "/data/selinawisco/kochild/five_fold_datasets/"
    data = f'{word_input_dir}test_fold_{fold}.csv'
    # data = pd.read_csv(f'{word_input_dir}test_fold_{fold}.csv') # test file
    # data = f'/home/selinawisco/whisper_evals/whisper-small-fold{fold}-42-eval.csv' # transcribed test file

    # 나뉜 음성 저장할 폴더 이름
    output_dir = f'/data/selinawisco/kochild/forced_aligned/fold_{fold}/human-aligned-fold-{fold}-test/'
    os.makedirs(output_dir, exist_ok=True)

    # 나뉜 음성에 대해 생성할 CSV 저장 경로
    csv_output_path = f'/data/selinawisco/kochild/forced_aligned/fold_{fold}/human_aligned_fold_{fold}_test.csv'

    # forced alignment 함수 호출
    forced_aligned_data = get_forced_alignment_data(data, output_dir, csv_output_path, test_time=False)

