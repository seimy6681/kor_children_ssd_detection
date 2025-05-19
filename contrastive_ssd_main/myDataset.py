# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from datasets import load_dataset
import torchaudio
import random
import numpy as np
import random

'''
스테레오 오디오 모노로 변환 및 sampling_rate 를 16000 으로 리샘플링
'''
@staticmethod
def load_mono_audio(path):
    waveform, sr = torchaudio.load(path)

    if waveform.ndim == 2 and waveform.shape[0] > 1:
        # Stereo → Mono
        waveform = waveform.mean(dim=0)
    elif waveform.ndim == 2 and waveform.shape[0] == 1:
        # Mono but shaped [1, T] → squeeze
        waveform = waveform.squeeze(0)
    elif waveform.ndim == 1:
        # Already fine
        pass
    else:
        raise ValueError(f"Unexpected waveform shape: {waveform.shape}")

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    
    return waveform


# ========== Data Preparation ==========
class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, audio_column,label_column, text_column, seed):
        self.data = dataframe
        self.audio_column = audio_column
        self.label_column = label_column
        self.text_column = text_column
        self.seed = seed
        
        self.correct = self.data[self.data[self.label_column] == 0].reset_index(drop=True)
        self.incorrect = self.data[self.data[self.label_column] == 1].reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)

    '''
    랜덤을 선정된 음성과 0/1이 같고 타겟단어도 같은 positive_audio 와, 0/1이 다른 negativ_audio 리턴
    '''
    def __getitem__(self, idx):
        # if random.random() < 0.5:
        #     anchor_df = self.correct
        #     negative_df = self.incorrect
        # else:
        #     anchor_df = self.incorrect
        #     negative_df = self.correct
        
        anchor_df = self.correct
        negative_df = self.incorrect
        
        # no target word matching between anchor, positive, negative
        # anchor_sample, positive_sample = random.sample(list(anchor_df[self.audio_column]),2)
        # negative_sample = random.sample(list(negative_df[self.audio_column]),1)
        
        anchor_row = anchor_df.sample(1).iloc[0]
        anchor_path = anchor_row[self.audio_column]
        anchor_text = anchor_row[self.text_column]
        
        pos_candidates = anchor_df[anchor_df[self.text_column]==anchor_text]
        pos_candidates = pos_candidates[pos_candidates[self.audio_column]!=anchor_path]
        
        if len(pos_candidates) == 0:
            positive_row = anchor_row
        else:
            positive_row = pos_candidates.sample(1).iloc[0]
            
        positive_path = positive_row[self.audio_column]
        
        neg_candidates = negative_df[negative_df[self.text_column]==anchor_text]
        if len(neg_candidates) == 0:
            negative_row = negative_df.sample(1).iloc[0] # if no negative sample with matching text, grab negative sample with different text
        else:
            negative_row = neg_candidates.sample(1).iloc[0]
        
        negative_path = negative_row[self.audio_column]
        
        anchor_audio = load_mono_audio(anchor_path)
        positive_audio = load_mono_audio(positive_path)
        negative_audio = load_mono_audio(negative_path)
        
        return anchor_audio, positive_audio, negative_audio

