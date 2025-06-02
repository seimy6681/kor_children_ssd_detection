# WhisperForConditionalGeneration 대신  models/ 안에 multitask_whisper_for_conditional_generation.py 모델을 불러와 asr finetuning 한다.
# https://huggingface.co/blog/fine-tune-whisper

import torch
import os
from datasets import load_dataset
from datasets import Audio
import debugpy
import wandb
from transformers import GenerationConfig
from multitask_trainer import MultitaskTrainer

debugpy.listen(5678)
print("waiting for debugger")
debugpy.wait_for_client()
# debugpy.breakpoint()
# print('break on this line')

num_runs = 1 # 연속으로 학습할 수
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CSV 경로 설정 --------------------------------------------------------------------------
DATA_PATH = '/data/selinawisco/kochild/five_fold_datasets/'
train = load_dataset("csv", data_files={"train": DATA_PATH + "test_fold_0_train.csv"}, delimiter=",")["train"]
# train = load_dataset("csv", data_files={"train": '/home/selinawisco/selina_main/asr/01_asr_train_sample.csv'}, delimiter=",")["train"]
test = load_dataset("csv", data_files={"test": DATA_PATH + "test_fold_0.csv"}, delimiter=",")["test"]
# ---------------------------------------------------------------------------------------

train = train.rename_column('human_text_jamo', 'sentence')
test = test.rename_column('human_text_jamo', 'sentence')
train = train.rename_column('speech_file', 'audio')
test = test.rename_column('speech_file', 'audio')

train = train.remove_columns(['disease_type', 'age', 'gender','subgroup', 'id', 'textgrid_text', 'target_text','asr_text','target_text_jamo','human_text'])
test = test.remove_columns(['disease_type', 'age', 'gender','subgroup', 'id', 'textgrid_text', 'target_text','asr_text','target_text_jamo','human_text'])

train = train.cast_column("audio", Audio(sampling_rate=16_000))
test = test.cast_column("audio", Audio(sampling_rate=16_000))

from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-small', language='Korean', task='transcribe')
# tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-large-v3-turbo', language='Korean', task='transcribe')


from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Korean", task="transcribe")
# processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo", language="Korean", task="transcribe")
# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
# processor.feature_extractor.chunk_length = None  # <-- disable default 30s padding

def prepare_dataset(batch):

    audio = batch['audio']
    features = feature_extractor(
        audio['array'], 
        sampling_rate=audio['sampling_rate'],
        return_attention_mask=True, # add attention_mask so it can be accessed for binary classifier,
        # padding="max_length",
        # truncation=True,
        # max_length=600,
    )
    
    batch['input_features']= features.input_features[0]
    batch['attention_mask']= features.attention_mask[0]
    
    # print(batch['input_features'].shape) # (128, 3000)
    # features = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate'])
    # batch['input_features'] = features.input_features
    batch['labels'] = tokenizer(batch['sentence']).input_ids
    # 0/1 mispronunciation 레이블 추가
    batch['binary_label'] = batch['new_label']

    return batch

train = train.map(prepare_dataset, num_proc=1)
test = test.map(prepare_dataset, num_proc=1)

print(test[0].keys())
from whisper_models.multitask_whisper_for_conditional_generation import MultitaskWhisperForConditionalGeneration
# from transformers import WhisperForConditionalGeneration
model = MultitaskWhisperForConditionalGeneration.from_pretrained('openai/whisper-small')
model.generation_config = GenerationConfig.from_pretrained("openai/whisper-small")
# model.generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3-turbo")
# print(model.generation_config)
model.generation_config.language = "korean"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
# model = torch.nn.DataParallel(model)

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

        # assert "binary_label" in features[0], "binary_label missing in eval batch features"
        # Add binary label to the batch
        if "binary_label" in features[0]:
            batch["binary_labels"] = torch.tensor([f["binary_label"] for f in features], dtype=torch.float) # creating a list of binary labels
        # Add attention mask to the batch
        if "attention_mask" in features[0]:
            attention_masks = [torch.tensor(f["attention_mask"], dtype=torch.float) for f in features]
            batch["attention_mask"] = torch.stack(attention_masks).float()
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

import evaluate
from sklearn.metrics import recall_score
metric = evaluate.load("cer")

def compute_metrics(pred):

    pred_ids, binary_logits = pred.predictions # get both asr logits and binary classification logits
    # pred_ids = pred.predictions
    label_ids, binary_labels = pred.label_ids # get both asr logits and binary classification logits
    
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    cer = metric.compute(predictions=pred_str, references=label_str)

    # binary classification metric (UAR)
    probs = torch.sigmoid(torch.tensor(binary_logits))
    preds = (probs > 0.5).long()

    binary_labels = torch.tensor(binary_labels).long()
    recall_0 = recall_score(binary_labels, preds, pos_label=0)
    recall_1 = recall_score(binary_labels, preds, pos_label=1)
    uar = 0.5 * (recall_0 + recall_1)

    return {"cer": cer, "uar": uar, "recall_0": recall_0, "recall_1": recall_1}


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
    name=f"multitask-whisper-seed-{seed}",
    config={
        "learning_rate": 3e-4,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "fp16": True,
        "max_steps": 5000,
        "seed": seed,
        "alpha" : 0.3,
    },
)
        
    training_args = Seq2SeqTrainingArguments(
        seed=seed,
        output_dir=f"/data/selinawisco/whisper_finetuning_asr/multitask-whisper-small-{seed}",  # change to a repo name of your choice
        save_total_limit=2,
        # per_device_train_batch_size=16,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        # gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=3e-4,
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
        # save_steps=1000,
        # eval_steps=5000,
        eval_steps=1000,
        logging_steps=25,
        dataloader_num_workers=4,
        # report_to=["tensorboard"],
        # load_best_model_at_end=False,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False,
        report_to=["wandb"]
    )

    # from transformers import Seq2SeqTrainer

    # trainer = Seq2SeqTrainer(
    trainer = MultitaskTrainer(
        args=training_args,
        model=model,
        train_dataset=train,
        eval_dataset=test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # trainer.train(resume_from_checkpoint="/data/selinawisco/whisper_finetuning_asr/whisper-large-v3-42/checkpoint-3000")
    trainer.train()
    tokenizer.save_pretrained(training_args.output_dir) 
    feature_extractor.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    
wandb.finish()


        