
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
import hangul_jamo
from whisper_models.phoneme_mtl_whisper_for_conditoinal_generation import PhonemeMtlWhisperForConditionalGeneration
# 테스트 csv 파일 로딩
TEST_FILE = '/data/selinawisco/kochild/five_fold_datasets/test_fold_0.csv'
df = pd.read_csv(TEST_FILE)
df=df.rename(columns={"speech_file":"audio"})

# 학습한 ASR whisper 모델 checkpoint 불러오기
MODEL_PATH = f"/data/selinawisco/whisper_finetuning_asr/phoneme_mtl_forced_alignment_test_fold_0_train_0.6_lr_5e5"


# # CUSTOMIZE ------------------------------------------------------------
MODE = 'human'  # 'human'
SEED = 42
# # ----------------------------------------------------------------------
TARGET_COL_NAME = f'{MODE}_text_jamo'
PRED_COL_NAME = f'asr_{MODE}_transcription'
WORD_CER_COL_NAME = f'word_{MODE}_CER'

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PhonemeMtlWhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model.to(device)
model.eval() # 평가 모드로 설정

# print(model)
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

print('whisper-small phoneme mtl transcribing')
df[PRED_COL_NAME] = df.apply(transcribe_audio, axis=1)

df[PRED_COL_NAME] = df[PRED_COL_NAME].apply(lambda text: re.sub(r'\[PAD\]', '', text))


# CER 계산
def calculate_row_cer(row):
    reference = row[TARGET_COL_NAME]
    hypothesis = row[PRED_COL_NAME]
    
    return jiwer.cer(reference, hypothesis)


df[TARGET_COL_NAME] = df[TARGET_COL_NAME].str.strip() # label 타겟 ground truth
df[PRED_COL_NAME] = df[PRED_COL_NAME].str.strip()
df['CER'] = df.apply(lambda x: calculate_row_cer(x), axis=1)
jiwer_avg_cer = df['CER'].mean()
print(f'{jiwer_avg_cer}')

TARGET_COL_NAME = 'target_text_jamo' #타겟단어 칼럼
PRED_COL_NAME = f'asr_{MODE}_transcription'

# # 전사 데이터 로딩 ------------------------------------------------------------------------
# test = f'/data/selinawisco/whisper_finetuning_asr/test_fold_0_train_lr_5e5.csv'
# df = pd.read_csv(test)

df['pred'] = (df['target_text_jamo']!=df['asr_human_transcription']).astype(int)
uar = recall_score(df['new_label'], df['pred'], average='macro') #
print(f'{uar=}')

tp = ((df['pred']==1) & (df['new_label'] ==1)).sum()
tn = ((df['pred']==0) & (df['new_label']==0)).sum()
fp = ((df['pred']==1) & (df['new_label'] ==0)).sum()
fn = ((df['pred']==0) & (df['new_label'] ==1)).sum()


# Output results
print(f"UAR(ASR): {uar} [{tp=}, {tn=}, {fp=}, {fn=}]")
from sklearn.metrics import classification_report
# df = pd.read_csv('0525_young_42_eval.csv')
print(classification_report(df['new_label'], df['pred']))

df.to_csv(f'/data/selinawisco/whisper_finetuning_asr/phoneme_mtl_forced_alignment_fold_0_train_0.6_lr_5e5.csv', index=False)
