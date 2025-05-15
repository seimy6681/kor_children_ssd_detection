from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import os
import re
import librosa
import pandas as pd
# import debugpy
import jiwer
import hangul_jamo
from sklearn.metrics import recall_score
import debugpy
from multitask_util.error_labels import phoneme_error_label
from models.attended_binary_multitask_wav2vec2 import AttendedBinaryMultitaskWav2Vec2ForCTC
from models.joint_attended_multitask_wav2vec2 import JointAttendedBinaryMultitaskWav2Vec2ForCTC
from models.binary_multitask_wav2vec2 import BinaryMultitaskWav2Vec2ForCTC
from models.phoneme_binary_multitask_wav2vec2 import PhonemeBinaryMultitaskWav2Vec2ForCTC
from models.age_classifier_multitask_wav2vec2 import AgeClassifierWav2Vec2ForCTC
from models.dann_age_classifier_wav2vec2 import AgeAdversarialWav2Vec2ForCTC
from models.age_loss_weighting_wav2vec2 import AgeLossWeightingWav2Vec2ForCTC
from models.age_embedding_wav2vec2 import AgeEmbeddingWav2Vec2ForCTC
import numpy as np

"""
MODEL_PATH : ASR 모델 경로
MODEL_TYPE: 불러올 모델의 키워드
    'baseline': Wav2Vec2, 
    'multitask': Multitask learning, 
    'fused': Multitask learning with Fused Attention,
    'age_embedding',
    'age_classifier',
    'age_loss'
TEST_FILE: 전사할 테스트 CSV 경로
RESULT_CSV_PATH: 전사결과 저장할 CSV 경로
"""


# CUSTOMIZE ------------------------------------------------------------
MODEL_TYPE = 'age_embedding' # 'baseline': Wav2Vec2, 'multitask': Multitask learning, 'fused': Multitask learning with Fused Attention
# MODEL_PATH = "/data/selinawisco/spring/0421_baseline_44" # 모델 경로
MODEL_PATH = "/data/selinawisco/spring/0508_age_label_embedding_44" # 모델 경로
MODE = 'human'  # 'human' pronunciation-based transcription -> 'human'
TEST_FILE = '/home/selinawisco/sel-hub/asr/r08.1_test.csv' 
RESULT_CSV_PATH = './result.csv'
# SEED = 45
# ----------------------------------------------------------------------

TARGET_COL_NAME = f'{MODE}_text_jamo'
PRED_COL_NAME = f'asr_{MODE}_transcription'

# 디버깅 -------------------------------------
# debugpy.listen(5678)
# print("waiting for debugger")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('break on this line')

# 디바이스 셋업  ----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 테스트 데이터 로딩 ------------------------------------------------------------------------
df = pd.read_csv(TEST_FILE)
# df = df[df['age']<=5]
# df.to_csv('young_r08.1_test.csv', index =False)
# df = pd.read_csv('young_r08.1_test.csv')

print(df)
# MODEL_TYPE 키워드 
if MODEL_TYPE == 'multitask':
    print('loading binary multitask wav2vec2 . . .')
    model = BinaryMultitaskWav2Vec2ForCTC.from_pretrained(MODEL_PATH)
elif MODEL_TYPE == 'fused':
    print('loading binary multitask wav2vec2 with fused attention . . .')
    model = AttendedBinaryMultitaskWav2Vec2ForCTC.from_pretrained(MODEL_PATH)
elif MODEL_TYPE == 'phoneme':
    print('loading phoneme binary multitask wav2vec2 . . .')
    model = PhonemeBinaryMultitaskWav2Vec2ForCTC.from_pretrained(MODEL_PATH)
elif MODEL_TYPE == 'phoneme_binary':
    print('loading phoneme binary multitask wav2vec2 . . .')
    model = PhonemeBinaryMultitaskWav2Vec2ForCTC.from_pretrained(MODEL_PATH)
elif MODEL_TYPE == 'age':
    print('loading age classifier multitask wav2vec2 . . .')
    model = AgeClassifierWav2Vec2ForCTC.from_pretrained(MODEL_PATH)
elif MODEL_TYPE == 'age_adversarial':
    print('loading age adversarial wav2vec2 . . .')
    model = AgeAdversarialWav2Vec2ForCTC.from_pretrained(MODEL_PATH)
elif MODEL_TYPE == 'age_loss':
    model = AgeLossWeightingWav2Vec2ForCTC.from_pretrained(MODEL_PATH)
elif MODEL_TYPE == 'age_embedding':
    model = AgeEmbeddingWav2Vec2ForCTC.from_pretrained(MODEL_PATH)
else: # baseline
    print('loading baseline wav2vec2 . . . ')
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
        
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model.to(device)
model.eval() # 평가 모드로 설정

print(model)
print('transcribing . . .')

vocab_dict = processor.tokenizer.get_vocab()
vocab_list = sorted(vocab_dict, key=vocab_dict.get)


# 전사하기
def transcribe_audio(row):
    file_path = row['audio']
    age_label = row['age']
    audio, sampling_rate = librosa.load(file_path, sr=16000)
    
    # Preprocess Audio Data
    input_values = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_values
    input_values = input_values.to(device)
    
    with torch.no_grad():
        if MODEL_TYPE != 'age_embedding':
            logits = model(input_values).logits
        elif MODEL_TYPE == 'age_embedding':
            age_label = torch.tensor(age_label).unsqueeze(0).type(torch.LongTensor)

            logits = model(input_values=input_values, age_label=age_label).logits
        if MODEL_TYPE == 'fused' or MODEL_TYPE == 'multitask':
            ac_logits = model(input_values).error_logits
        if MODEL_TYPE == 'phoneme' or MODEL_TYPE == 'phoneme_binary':
            phoneme_logits = model(input_values).phoneme_error_logits
        
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    tokens_to_remove = ['[PAD]', '[UNK]', '|', '</s>']
    for token in tokens_to_remove:
        transcription = transcription.replace(token, '')
        
    asr_probs = torch.nn.functional.softmax(logits, dim=-1)
    asr_confidence = asr_probs.max(dim=-1).values.mean(dim=-1)
    
    if MODEL_TYPE == 'fused' or MODEL_TYPE == 'multitask':
        
        normal_logit = ac_logits[0,0].cpu().item()
        ssd_logit = ac_logits[0,1].cpu().item()
        PRED_by_AC = 0 if normal_logit > ssd_logit else 1
        ac_confidence= torch.sigmoid(ac_logits[:,1]) # grab the second logit for class 1
        
        return transcription, normal_logit, ssd_logit, PRED_by_AC, asr_confidence.cpu().item(), ac_confidence.cpu().item()
    
    if MODEL_TYPE == 'phoneme' or MODEL_TYPE == 'phoneme_binary':
        # correct_logits = phoneme_logits[0,0].cpu().item()
        # incorrect_logits = phoneme_logits[0,1].cpu().item()
        pred = torch.argmax(phoneme_logits, dim=-1)
        # print(phoneme_logits)
        # print(pred)
        phoneme_preds = str(pred.cpu().numpy())  # Ensure it's on CPU

        return transcription, phoneme_preds, asr_confidence.cpu().item()
        
    return transcription


if MODEL_TYPE == 'baseline' or MODEL_TYPE == 'age' or MODEL_TYPE == 'age_adversarial' or MODEL_TYPE == 'age_loss' or MODEL_TYPE=='age_embedding': # TODO
    df[PRED_COL_NAME] = df.apply(transcribe_audio, axis=1)
    
if MODEL_TYPE == 'fused' or MODEL_TYPE == 'multitask':
    df[[PRED_COL_NAME, 'normal_logit', 'ssd_logit', 'pred_by_ac','asr_confidence', 'ac_confidence']] = df.apply(
        lambda row: pd.Series(transcribe_audio(row)), axis=1
    )
if MODEL_TYPE == 'phoneme' or MODEL_TYPE == 'phoneme_binary':
    df[[PRED_COL_NAME, 'phoneme_preds','asr_confidence']] = df.apply(
        lambda row: pd.Series(transcribe_audio(row)), axis=1
    )  
    
df[PRED_COL_NAME] = df[PRED_COL_NAME].apply(lambda text: re.sub(r'\[PAD\]', '', text))


# CER 계산
def calculate_row_cer(row):
    reference = row[TARGET_COL_NAME]
    hypothesis = row[PRED_COL_NAME]
    
    return jiwer.cer(reference, hypothesis)

df['CER'] = df.apply(lambda x: calculate_row_cer(x), axis=1)
jiwer_avg_cer = df['CER'].mean()
print(f'CER = {jiwer_avg_cer}')

# df.to_csv(f'/home/selinawisco/finetuning_baseline_43_1', index=False)

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
   

# 전사 결과 csv 로 저장시 경로 설정:
df.to_csv(RESULT_CSV_PATH, index=False)
         
# UAR 계산
sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

uar = (sensitivity + specificity) / 2

# Output results
print(f"UAR(ASR): {uar} [{true_positives=}, {true_negatives=}, {false_positives=}, {false_negatives=}]")
from sklearn.metrics import classification_report
# df = pd.read_csv('0525_young_42_eval.csv')
print(classification_report(df['new_label'], df['pred_by_ASR']))
if MODEL_TYPE == 'fused' or MODEL_TYPE == 'multitask':
    uar_by_ac = recall_score(df['new_label'],df['pred_by_ac'], average='macro')
    print(f'UAR(AC): {uar_by_ac}')

if MODEL_TYPE == 'phoneme':
     print('TODO')
  


