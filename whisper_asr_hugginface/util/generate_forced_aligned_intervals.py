from tqdm import tqdm
from forced_aligner import forced_alignment
import json 
import pandas as pd
# 기존 단어단위 csv 에 각 음성별로 음소단위 타임스탭을 dictionary로 새 칼럼에 저장하는 함수
def generate_phoneme_interval_column(df, audio_col="audio", transcript_col="human_text_jamo", output_col="phoneme_intervals"):
    intervals = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row[audio_col]
        transcript = row[transcript_col]
        
        segments = forced_alignment(audio_path, transcript)
        intervals.append(json.dumps(segments, ensure_ascii=False))
        
    df[output_col] = intervals
    return df

# 실행 예시:
# (1) approach 1: saving the phoneme level intervals for each sample  
for fold in range(0,5):
    
    df = pd.read_csv(f'/data/selinawisco/kochild/five_fold_datasets/test_fold_{fold}.csv')
    result = generate_phoneme_interval_column(df)
    result.to_csv(f'/data/selinawisco/five_fold_with_phoneme_intervals/forced_alignment_test_fold_{fold}.csv', index=False)
    