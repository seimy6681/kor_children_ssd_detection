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

df.to_csv(f'/home/selinawisco/selina_main/asr/whisper-small-hi-45-eval.csv', index=False)
