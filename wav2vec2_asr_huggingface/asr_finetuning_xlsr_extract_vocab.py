# Korean Dictionary 생성 ----------------------------------------- #

import pandas as pd
import debugpy
import torch
import os
import jiwer
# import torchaudio


# 허깅페이스 데이터셋으로 변환 ------------------------
from datasets import load_dataset, load_metric, Audio

train = load_dataset("csv", data_files={"train": "/data/selinawisco/kochild/five_fold_datasets/test_fold_0_train.csv"}, delimiter=",")["train"]
# train = load_dataset("csv", data_files={"train": '/home/selinawisco/selina_main/asr/01_asr_train_sample.csv'}, delimiter=",")["train"]
test = load_dataset("csv", data_files={"test": "/data/selinawisco/kochild/five_fold_datasets/test_fold_0.csv"}, delimiter=",")["test"]

# 데이터셋 Transcript에 나오는 한글 캐릭터 추출 -------------------
def extract_all_chars(batch):
    # filter out none entry
    # transcripts = [text for text in batch['transcript'] if text is not None]
    # all_text = " ".join(transcripts) # connect all strings with space
    all_text = " ".join(batch['target_text_jamo']) # connect all strings with space
    all_human_text = " ".join(batch['human_text_jamo'])
    
    # print(all_text)
    vocab = list(set(all_text)) # remove duplicates
    human_vocab = list(set(all_human_text)) 
    vocab = set(vocab + human_vocab)
    
    return {'vocab': [vocab], "all_text": [all_text]} # create dictionary

vocab_train = train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train.column_names)
vocab_test = train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=test.column_names)

vocab_list = list(set(vocab_train['vocab'][0]) | set(vocab_test['vocab'][0]))
# print(vocab_list)

vocab_dict = {v: k for k, v in enumerate(vocab_list)} # create dictionary with the enumerate value as index

# if " " not in vocab_dict:
#     vocab_dict[" "] = len(vocab_dict)  # Assign a new index to the space character
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict['[UNK]'] = len(vocab_dict)
vocab_dict['[PAD]'] = len(vocab_dict)

# print(vocab_dict)
# print(len(vocab_dict))

# Vocabulary JSON 으로 저장 ------------------------
import json
with open('vocab_jrnl.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)