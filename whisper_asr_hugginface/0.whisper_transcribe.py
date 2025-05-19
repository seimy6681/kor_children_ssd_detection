from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import os
import re
import librosa
import pandas as pd
import debugpy
import jiwer
from sklearn.metrics import recall_score
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from sklearn.metrics import recall_score

# 테스트 csv 파일 로딩
df = pd.read_csv('/home/selinawisco/hdd/korean_asr/r08.1_test.csv')

# 학습한 ASR whisper 모델 checkpoint 불러오기
MODEL_PATH = f"/home/selinawisco/selina_main/asr/whisper-small-hi-45"

# CUSTOMIZE ------------------------------------------------------------
MODE = 'human'  # 'human'
TEST_FILE = '/home/selinawisco/sel-hub/asr/r08.1_test.csv'
SEED = 45
# ----------------------------------------------------------------------
TARGET_COL_NAME = f'{MODE}_text_jamo'
PRED_COL_NAME = f'asr_{MODE}_transcription'
WORD_CER_COL_NAME = f'word_{MODE}_CER'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model.to(device)
model.eval() # 평가 모드로 설정

print(model)
# # Transcribe the Audio
def transcribe_audio(row):
    file_path = row['audio']
    audio, sampling_rate = librosa.load(file_path, sr=16000)
    # PreProcess Audio Data
    input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
    input_features = input_features.to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_features)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # transcription = re.sub(r'\[PAD\]', '', transcription[0])
    tokens_to_remove = ['[PAD]', '[UNK]', '|'] # 전사 외 스페셜 토큰 제거
    for token in tokens_to_remove:
        transcription = transcription.replace(token, '')
    

    return transcription


df[PRED_COL_NAME] = df.apply(transcribe_audio, axis=1)

df[PRED_COL_NAME] = df[PRED_COL_NAME].apply(lambda text: re.sub(r'\[PAD\]', '', text))


# CER 계산
def calculate_row_cer(row):
    reference = row[TARGET_COL_NAME]
    hypothesis = row[PRED_COL_NAME]
    
    return jiwer.cer(reference, hypothesis)

df['CER'] = df.apply(lambda x: calculate_row_cer(x), axis=1)
jiwer_avg_cer = df['CER'].mean()
print(f'{jiwer_avg_cer}')


TARGET_COL_NAME = 'target_text_jamo' #타겟단어 칼럼
PRED_COL_NAME = f'asr_{MODE}_transcription'

# # 전사 데이터 로딩 ------------------------------------------------------------------------
# test = f'/home/selinawisco/finetuning_baseline_43_1'
# df = pd.read_csv(test)

# 단어 추출 (whitespace 제거)
df[TARGET_COL_NAME] = df[TARGET_COL_NAME].str.strip() # label 타겟 ground truth
df[PRED_COL_NAME] = df[PRED_COL_NAME].str.strip()

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

fp_errors = {}
fn_errors = {}
tp = []

fn_rows = []

for idx,(_, row) in enumerate(df.iterrows()):
    target_jamo = row[TARGET_COL_NAME]
    # asr = row['asr_transcription']
    asr = row[PRED_COL_NAME]
    new_label = row['new_label']
    
    if new_label == 1: # 레이블 상 단어 발음이 틀렸을 시
        if target_jamo != asr: # asr이 타겟단어와 다르기만 하면 True Positive 장애진단 (정답)
            true_positives += 1
            tp.append(row)
            df.at[idx, 'pred_by_ASR'] = 1
            
        else:
            false_negatives += 1 # asr과 타겟단어가 일치하면 false negative 정상진단 (오류)   
            dict_key = row['target_text'] + "_" + str(idx)         
            fn_errors[dict_key] = (row['human_text'])
            fn_rows.append(row)
            df.at[idx, 'pred_by_ASR'] = 0
            
            
    elif new_label == 0: # 레이블 상 단어 발음이 정상일 시
        if target_jamo == asr: # asr과 타겟단어가 같으면 정상진단 (정답)
            true_negatives += 1
            df.at[idx, 'pred_by_ASR'] = 0
            
        else:
            false_positives += 1 # asr 과 타겟단어가 다르면 장애진단 (오류)
            dict_key_2 = row['target_text'] + "_" + str(idx)
            fp_errors[dict_key_2] = hangul_jamo.compose(row['asr_human_transcription'])
            df.at[idx, 'pred_by_ASR'] = 1
   

        
# UAR 계산
sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

uar = (sensitivity + specificity) / 2
# uar = recall_score(df['new_label'], df['asr_human_transcription'], average='macro') #
# print(f'{uar=}')

# Output results
print(f"UAR(ASR): {uar} [{true_positives=}, {true_negatives=}, {false_positives=}, {false_negatives=}]")
from sklearn.metrics import classification_report
# df = pd.read_csv('0525_young_42_eval.csv')
print(classification_report(df['new_label'], df['pred_by_ASR']))

df.to_csv(f'/home/selinawisco/selina_main/asr/whisper-small-hi-45-eval.csv', index=False)
