from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import librosa
import torch
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os
import soundfile as sf
import pandas as pd 
from tqdm import tqdm
import json 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/data/selinawisco/spring/0421_baseline_44"
model = Wav2Vec2ForCTC.from_pretrained(model_path).eval()
processor = Wav2Vec2Processor.from_pretrained(model_path)

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
    audio, sr = librosa.load(audio_path, sr=16000)
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
                "phoneme" : transcript[idx],
                "start_sample" : start_sample,
                # "start_sample_whisper" : start_sample // 320, # for whisper
                "end_sample": end_sample,
                # "end_sample_whisper": end_sample // 320, # for whisper
            })
            
    return interval_data

def split_phoneme_segments(df,
                           audio_col="speech_file",
                           transcript_col="human_text_jamo",
                        #    target_text_col="target_text_jamo",
                           save_dir="./phoneme_segments",
                           sample_rate=16000):
    os.makedirs(save_dir, exist_ok=True)
    segment_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row[audio_col]
        transcript = row[transcript_col]
        

        audio, sr = librosa.load(audio_path, sr=sample_rate)

        segments = forced_alignment(audio_path, transcript)

        for i, seg in enumerate(segments):
            start = seg["start_sample"]
            end = seg["end_sample"]
            phoneme = seg["phoneme"]

            segment_audio = audio[start:end]
            file_name = f"{os.path.splitext(os.path.basename(audio_path))[0]}_{i}_{phoneme}.wav"
            save_path = os.path.join(save_dir, file_name)

            sf.write(save_path, segment_audio, sample_rate)

            segment_data.append({
                "orig_audio": audio_path,
                "segment_audio": save_path,
                "phoneme": phoneme,
                "start_sample": start,
                "end_sample": end,
                "human_text_jamo": transcript,
                "target_text_jamo": row["target_text"],
                "disease_type": row["disease_type"],
                "age": row["age"],
                "gender": row["gender"],
                "id":row["id"],
                "subgroup": row["subgroup"]
            })
    return pd.DataFrame(segment_data)


# 음소단위 음성 저장 및 csv 생성 ------------------------------------------------------------ #
fold = 0
save_dir = "/data/selinawisco/kochild/phoneme_segments_audio/"
csv_save_path = "/data/selinawisco/kochild/phoneme_segments_datasets/"
os.makedirs(csv_save_path, exist_ok=True)
train = pd.read_csv(f'/data/selinawisco/kochild/five_fold_datasets/test_fold_{fold}_train.csv')
test = pd.read_csv(f'/data/selinawisco/kochild/five_fold_datasets/test_fold_{fold}.csv')

train_phoneme = split_phoneme_segments(train, save_dir=save_dir)
train_phoneme.to_csv(f'/data/selinawisco/kochild/phoneme_segments_datasets/test_fold_{fold}_train_phoneme.csv', index=False)

test_phoneme = split_phoneme_segments(test, save_dir=save_dir)
test_phoneme.to_csv(f'{csv_save_path}test_fold_{fold}_phoneme.csv', index=False)
# --------------------------------------------------------------------------------------- #


# 음성별 음소단위 타임스탭 칼럼 저장하는 함수
def generate_phoneme_interval_column(df, audio_col="speech_file", transcript_col="human_text_jamo", output_col="phoneme_intervals"):
    intervals = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row[audio_col]
        transcript = row[transcript_col]
        
        segments = forced_alignment(audio_path, transcript)
        intervals.append(json.dumps(segments, ensure_ascii=False))
        
    df[output_col] = intervals
    return df
    

# (1) approach 1: saving the phoneme level intervals for each sample  
# df = pd.read_csv('/data/selinawisco/kochild/five_fold_datasets/test_fold_0.csv')
# result = generate_phoneme_interval_column(df)
# result.to_csv('./forced_alignment_test_fold_0_1.csv', index=False)


                        