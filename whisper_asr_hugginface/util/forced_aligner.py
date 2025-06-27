# https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
# Wav2Vec2 CTC 모델을 이용해 forced alignment 구현한 pytorch 튜토리얼
# 2025.06.16

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import torch
from dataclasses import dataclass
import os
import soundfile as sf
import pandas as pd 
from tqdm import tqdm
from itertools import chain
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# (1) 강제정렬에 필요한 Wav2Vec2-CTC 모델을 불러오기 ================================================================ #
# note: FOLD 를 제외한 나머지 4개의 folds로 학습한 Wav2Vec2-CTC 모델 체크포인트를 불러옵니다 (학습코드 위치: asr_ssd_main/run.sh)
model_path = "/data/selinawisco/wav2vec2-word-level-asr-baseline_five_fold_datasets/test_fold_0_train.csv_42" # 저장된 모델 경로
model = Wav2Vec2ForCTC.from_pretrained(model_path).eval()
processor = Wav2Vec2Processor.from_pretrained(model_path)
# ========================================================================================================== #


# PYTORCH Forced Alignment Algorithm ======================================================================= #
def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]

# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def frame_to_sample(frame_idx, total_frames, total_samples):
    return int((frame_idx / total_frames) * total_samples)

def forced_alignment(audio_path, transcript):
    
    audio, sr = librosa.load(audio_path, sr=16000) # 음성 로드
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    
    with torch.no_grad():
        inputs = processor(input_values.squeeze(), return_tensors="pt", sampling_rate=16000)
        logits = model(**inputs).logits
        
        vocab_dict = processor.tokenizer.get_vocab()  # {token: id}
        tokens = [vocab_dict[c] for c in transcript if c in vocab_dict]

        # print(logits.shape)
        # emissions = torch.log_softmax(logits, dim=-1)[0]
        emissions = torch.log_softmax(logits / 1.5, dim=-1)[0]
        
        trellis = get_trellis(emissions, tokens)
        path = backtrack(trellis, emissions, tokens)
        
        segments = merge_repeats(path, transcript)
               
        total_frames = emissions.shape[1] if emissions.ndim == 3 else emissions.shape[0]
        total_samples = len(audio) 
        
        interval_data = []
        for idx, seg in enumerate(segments):
            
            start_sample = frame_to_sample(seg.start, total_frames, total_samples)
            end_sample = frame_to_sample(seg.end, total_frames, total_samples)

            interval_data.append({
                "fa_label" : transcript[idx],
                "start_sample" : start_sample,
                # "start_sample_whisper" : start_sample // 320, # for whisper
                "end_sample": end_sample,
                # "end_sample_whisper": end_sample // 320, # for whisper
            })
            
    return interval_data

# PYTORCH Forced Alignment Algorithm ============================================================== #
# end of PYTORCH TUTORIAL code -------------------------------------------------------------------- # 


# CSV 단위 Forced Alignment 실행 관련 함수 ============================================================= #
def split_phoneme_segments_row(row,
                                audio_col="audio",
                                # human_col="human_h2j",
                                human_col="human_text_jamo",
                                # target_col="target_h2j",
                                target_col="target_text_jamo",
                                # asr_col="asr_h2j",
                                asr_col="asr_human_transcription",
                                save_dir="./phoneme_segments",
                                sample_rate=16000,
                                test_time=False):
    
    os.makedirs(save_dir, exist_ok=True)
    segment_data = []

    audio_path = row[audio_col]
    human_transcript = row[human_col]
    target_transcript = row[target_col]
    # asr_transcript = row[asr_col]

    audio, sr = librosa.load(audio_path, sr=sample_rate)

    try:
        if test_time: # at test time, human_text is not available -> use target text for alignment
            segments = forced_alignment(audio_path, target_transcript)
        else: # at training time, use human_text for forced alignment
            segments = forced_alignment(audio_path, human_transcript)
            
    except Exception as e:
        print(f"Failed alignment for {e}")
        return []  # or np.nan if you want to flag it

    for i, seg in enumerate(segments):
        start = seg["start_sample"]
        end = seg["end_sample"]
        fa_label = seg["fa_label"]

        # get corresponding target phoneme at same position
        target_phoneme = target_transcript[i] if i < len(target_transcript) else ""
        human_phoneme = human_transcript[i] if i < len(human_transcript) else ""
        # asr_phoneme = asr_transcript[i] if i < len(asr_transcript) else ""
        
        segment_audio = audio[start:end]
        file_name = f"{os.path.splitext(os.path.basename(audio_path))[0]}_{i}_{fa_label}.wav"
        save_path = os.path.join(save_dir, file_name)

        sf.write(save_path, segment_audio, sample_rate)

        segment_data.append({
            "orig_audio": audio_path,
            "segment_audio": save_path,
            "fa_phoneme_label": fa_label,
            "phoneme_idx": i, 
            "target_phoneme": target_phoneme,
            "human_phoneme": human_phoneme,
            # "asr_phoneme": asr_phoneme,
            "start_sample": start,
            "end_sample": end,
            "human_text_jamo": human_transcript,
            "target_text_jamo": target_transcript,
            # "asr_human_transcription": asr_transcript,
            "disease_type": row["disease_type"],
            "age": row["age"],
            "gender": row["gender"],
            "id": row["id"],
            "subgroup": row["subgroup"]
        })

    return segment_data  # list of dicts

def get_forced_alignment_data(data_path, save_dir_audio, save_path_csv, test_time=False):
    """
    This function generates phoneme level dataset (audio, csv) given word level dataset using fored alignment with wav2vec2. (model is pre-loaded at the top of the code)
    ::params::
    data_path: path to data to force align (word level dataset)
    save_dir_audio: path to save the force aligned audio
    save_dir_csv: path save the force aligned audio csv
    test_time: if set to True, target_text is used for forced alignment; else, human_text is used.
    """
    data = pd.read_csv(data_path)

    print("forced aligning ", os.path.basename(data_path), ". . .")
    tqdm.pandas()
    
    segments = list(chain.from_iterable(
        data.progress_apply(lambda row: split_phoneme_segments_row(row, save_dir=save_dir_audio,test_time=test_time), axis=1)
    ))
    print("forced aligned audio saved to ", save_dir_audio)
    
    # turn dictionary entries into csv
    result_df = pd.DataFrame(segments)
    result_df.to_csv(save_path_csv, index=False)
    print("forced alignment csv saved to ", save_path_csv)
    
    return result_df



## 실행 코드 =================================================================================== #####
# 데이터셋 fold 지정
fold = 0

# (2) 강제정렬에 필요한 input word-level CSV 데이터 읽어오기
input_data_root_dir = "/data/selinawisco/kochild/five_fold_datasets/" # input 데이터 루트 디렉토리

# input_data_csv = f'{input_data_root_dir}test_fold_{fold}_train.csv' # train file
input_data_csv = f'{input_data_root_dir}test_fold_{fold}.csv' # test file
# data = f'/home/selinawisco/whisper_evals/whisper-small-fold{fold}-42-eval.csv' # transcribed test file

# (2) 나뉜 음소단위 음성이 저장될 폴더 이름 정의 및 생성
output_dir = f'/data/selinawisco/kochild/forced_aligned/fold_{fold}/human-aligned-fold-{fold}-test/'
os.makedirs(output_dir, exist_ok=True)
# (3) 나뉜 음소단위 음성에 대한 생성할 CSV 이름
csv_output_path = f'{output_dir}human_aligned_fold_{fold}_test.csv'

# (4) forced alignment 함수 호출
forced_aligned_data = get_forced_alignment_data(input_data_csv, output_dir, csv_output_path, test_time=False) # test_time=False for human_aligned

## 실행 코드 =================================================================================== #####
