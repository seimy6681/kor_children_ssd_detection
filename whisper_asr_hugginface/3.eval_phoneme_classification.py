
from whisper_models.whisper_for_phoneme_classification import WhisperForPhonemeClassification
from transformers import WhisperProcessor
import librosa
import torch
import json
import pandas as pd
from util.extract_vocab import extract_vocab
from util.add_target_phoneme_column import get_phoneme_at_index
from tqdm import tqdm
from sklearn.metrics import classification_report
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test = pd.read_csv(f'/data/selinawisco/kochild/forced_aligned/fold_4/human_aligned_fold_4_test.csv')

# loading model checkpoint, feature_extractor
MODEL_PATH = "/data/selinawisco/phoneme-classification-fold-1-2-3-whisper-small-42"
model = WhisperForPhonemeClassification.from_pretrained(MODEL_PATH)
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
feature_extractor = processor.feature_extractor

model = model.eval().to(device)
model_config = model.config

# # load the saved phoneme_to_id dictionary from model directory
# with open(f"{MODEL_PATH}/phoneme_to_id.json", "r") as f:
#     phoneme_to_id = json.load(f)

def classify_phoneme(row):
    
    audio, sr = librosa.load(row['segment_audio'], sr=16000)
    # input_features = processor(audio.squeeze(), sampling_rate=sr,return_tensors="pt")
    input_features = feature_extractor(audio, sampling_rate=sr,return_tensors="pt").input_features.to(device)
    
    with torch.no_grad():
        logits = model(input_features).logits
        pred_id = logits.argmax(dim=1).item()
    
    # return id_to_phoneme[pred_id]
    return model_config.id2label[pred_id]

tqdm.pandas()

# 각 음소 phoneme classification 으로 분류해서 새 칼럼에 저장
test['phoneme_pred'] = test.progress_apply(lambda row: classify_phoneme(row), axis=1)
test.to_csv('./phoneme_classification_fold_4_eval.csv', index=False)

print(classification_report(test['fa_phoneme_label'],test['phoneme_pred']))

# # phoneme level dataset에 word level asr 전사 정보 추가
# # asr_dict = df.set_index('audio')['asr_human_transcription'].to_dict()
# # test['asr_human_transcription'] = test['orig_audio'].map(asr_dict)

# # 해당 인덱스에 해당하는 ASR 전사 음소 뽑아서 저장
# test['asr_phoneme'] = test.progress_apply(lambda row: get_phoneme_at_index(row, 'asr'), axis=1)
# test.to_csv('./phoneme_classification_fold_0_eval.csv', index=False)

# # 타겟 음소와 asr 음소가 매치하지 않는 것중
# test = [test['target_phoneme']==test['asr_phoneme']]
# # 타겟 음소와 phoneme 분류 결과 같은 음성 갯수는?
# print(len(test[test['target_phoneme']==test['phoneme_pred']]))




    