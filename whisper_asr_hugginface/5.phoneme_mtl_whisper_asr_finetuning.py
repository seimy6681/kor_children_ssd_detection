# forced alignment 구간으로 음소별로 encoder_hidden_states를 자른 후 해당구간에 자모를 classify 하는 phoneme classification MTL 모델
# https://huggingface.co/blog/fine-tune-whisper

import torch
import os
from datasets import load_dataset
from datasets import Audio
import debugpy
import wandb
import pathlib
import re
import json
import pandas as pd
from whisper_models.phoneme_mtl_whisper_for_conditoinal_generation import PhonemeMtlWhisperForConditionalGeneration
from transformers import WhisperConfig

# debugpy.listen(5678)
# print("waiting for debugger")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('break on this line')

num_runs = 1 # 연속으로 학습할 수
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# /data/selinawisco/phoneme_mixing/mixed_test_fold_verified_age_and_correct_v1.csv'
train_file_path = "/data/selinawisco/kochild/forced_alignment_fold_0_train_1.csv"
train_filename = pathlib.Path(train_file_path).stem
print("training with ", train_filename)

# CSV 경로 설정 --------------------------------------------------------------------------
train = load_dataset("csv", data_files={"train": train_file_path}, delimiter=",")["train"]
# train = load_dataset("csv", data_files={"train": "/data/selinawisco/kochild/five_fold_datasets/test_fold_4_train.csv"}, delimiter=",")["train"]
test = load_dataset("csv", data_files={"test": "/data/selinawisco/kochild/forced_alignment_test_fold_0_1.csv"}, delimiter=",")["test"]
# ---------------------------------------------------------------------------------------

train = train.rename_column('human_text_jamo', 'sentence')
test = test.rename_column('human_text_jamo', 'sentence')

train = train.remove_columns(['disease_type', 'age', 'gender','subgroup', 'id', 'textgrid_text', 'target_text', 'new_label','target_text_jamo','human_text'])
test = test.remove_columns(['disease_type', 'age', 'gender','subgroup', 'id', 'textgrid_text', 'target_text', 'new_label','target_text_jamo','human_text'])


def clean_jamo(example):
    exclude_chars = r"[\[\]\|\?]"
    example["sentence"] = re.sub(exclude_chars, "", example["sentence"])
    return example

# Apply to Hugging Face dataset
train = train.map(clean_jamo)
test = test.map(clean_jamo)

train = train.cast_column("audio", Audio(sampling_rate=16_000))
test = test.cast_column("audio", Audio(sampling_rate=16_000))

train = train.filter(lambda x: x['phoneme_intervals_label'] is not None)
test = test.filter(lambda x: x['phoneme_intervals_label'] is not None)

from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-small', language='Korean', task='transcribe')


from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Korean", task="transcribe")
# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

MAX_TEXT_LEN = 16 # human_text_jamo 전체 최대 길이

import json

#jamo_to_id 딕셔너리 최초 생성
# def jamo_map(df, text_column):
#     label_column = df[text_column]
#     jamos = ''.join(label_column)
#     jamo_set = sorted(set(jamos))
#     exclude = {'[', ']', '?', '|', ' ', ''}
#     filtered_jamos = [ch for ch in jamo_set if ch not in exclude]
#     jamo_dict = {char :idx for idx, char in enumerate(filtered_jamos)}
#     return jamo_dict

# train_df = train.to_pandas()
# test_df = test.to_pandas()
# dataset = pd.concat([train_df,test_df])

# jamo_to_id = jamo_map(dataset, 'sentence')

# with open("jamo_to_id.json", "w", encoding="utf-8") as f:
#     json.dump(jamo_to_id, f, ensure_ascii=False, indent=4)
    
with open("jamo_to_id.json", "r", encoding="utf-8") as f:
    jamo_to_id = json.load(f)
    
print(len(jamo_to_id))
print(jamo_to_id)

def prepare_dataset(batch):

    audio = batch['audio']
    batch['input_features'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate']).input_features[0]
    batch['labels'] = tokenizer(batch['sentence']).input_ids
    
    EXCLUDE_TOKENS = {"[", "]", "|", "?", " ", ""}  # Add more if needed
    segments = json.loads(batch['phoneme_intervals_label'])
        
    intervals = []
    jamos = []
    
    for seg in segments:
        if seg["phoneme"] in EXCLUDE_TOKENS:
            continue # skip over invalid phoneme tokens
        intervals.extend([seg['start_sample_whisper'],seg['end_sample_whisper']]) # adding start sample, end sample
        jamos.append(jamo_to_id[seg['phoneme']]) # 자모를 id 로 변환
        
    # while len(jamos) < MAX_TEXT_LEN: # 16
    #         jamos.append(-100)
    # while len(intervals) < MAX_TEXT_LEN * 2:
    #     intervals.append(-100)
    
    # all_intervals.append(intervals)
    # all_jamos.append(jamos)
        
    batch['intervals'] = intervals
    batch['jamos'] = jamos
    return batch

train = train.map(prepare_dataset, num_proc=1, load_from_cache_file=True)
test = test.map(prepare_dataset, num_proc=1, load_from_cache_file=True)

train = train.remove_columns('phoneme_intervals_label')
test = test.remove_columns('phoneme_intervals_label')

from transformers import WhisperForConditionalGeneration
config = WhisperConfig.from_pretrained("openai/whisper-small")
config.num_jamos = len(jamo_to_id)
config.alpha = 0.3
model = PhonemeMtlWhisperForConditionalGeneration.from_pretrained('openai/whisper-small', config=config)
model.generation_config.language = "korean"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract the raw input_features list for padding
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad the labels separately
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Mask padding tokens in labels
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove the decoder start token if all labels start with it
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # Add labels to the batch
        batch["labels"] = labels
        
        # 3. Stack intervals
        intervals = [torch.tensor(f["intervals"], dtype=torch.long) for f in features]
        intervals = torch.nn.utils.rnn.pad_sequence(intervals, batch_first=True, padding_value=-100)
        batch["phoneme_intervals"] = intervals

        phonemes = [torch.tensor(f["jamos"], dtype=torch.long) for f in features]
        phonemes = torch.nn.utils.rnn.pad_sequence(phonemes, batch_first=True, padding_value=-100)
        batch["phoneme_targets"] = phonemes

        # intervals = [torch.tensor(feature["intervals"], dtype=torch.long) for feature in features]
        # batch["phoneme_intervals"] = torch.stack(intervals)

        # # 4. Stack phoneme labels
        # phonemes = [torch.tensor(feature["jamos"], dtype=torch.long) for feature in features]
        # batch["phoneme_targets"] = torch.stack(phonemes)


        return batch

    

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

import evaluate
metric = evaluate.load("cer")

def compute_metrics(pred):

    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    cer = metric.compute(predictions=pred_str, references=label_str)
    
    return {"cer": cer}

from transformers import Seq2SeqTrainingArguments


for seed in range(42, 42 + 1): 
    print(f"Running training with seed {seed}")

    # Set the new seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
        
    wandb.init(
    project="whisper-finetuning",
    name=f"whisper-seed-{seed}",
    config={
        "learning_rate": 3e-4,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "fp16": True,
        "max_steps": 5000,
        "seed": seed,
        "train_file": train_filename,
    },
)
        
    training_args = Seq2SeqTrainingArguments(
        seed=seed,
        output_dir=f"/data/selinawisco/whisper_finetuning_asr/phoneme_mtl_{train_filename}_{config.alpha}_lr_5e5",  # change to a repo name of your choice
        # output_dir=f"/data/selinawisco/whisper_finetuning_asr/whisper-small-fold-4-{seed}",  # change to a repo name of your choice
        save_total_limit=2,
        per_device_train_batch_size=8,
        # per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        # gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=5e-5,
        warmup_steps=500,
        max_steps=5000,
        # num_train_epochs=30,
        gradient_checkpointing=False,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        # per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=5000,
        logging_steps=25,
        dataloader_num_workers=4,
        # report_to=["tensorboard"],
        load_best_model_at_end=False,
        metric_for_best_model="cer",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False,
        report_to=["wandb"]
    )

    from transformers import Seq2SeqTrainer

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train,
        eval_dataset=test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    tokenizer.save_pretrained(training_args.output_dir) 
    feature_extractor.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    # trainer.evaluate()
    
wandb.finish()


        