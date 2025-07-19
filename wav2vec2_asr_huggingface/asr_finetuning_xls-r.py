import pandas as pd
import debugpy
import torch
import os
import jiwer
# import torchaudio

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# 디바이스 ------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("device: ", device)
print(torch.cuda.nccl.version())
print(torch.version.cuda)
# 디버깅 -------------------------------------

# debugpy.listen(5678)
# print("waiting for debugger")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('break on this line')

# 허깅페이스 데이터셋으로 변환 ------------------------
from datasets import load_dataset, load_metric, Audio

train = load_dataset("csv", data_files={"train": "/data/selinawisco/kochild/five_fold_datasets/test_fold_0_train.csv"}, delimiter=",")["train"]
test = load_dataset("csv", data_files={"test": "/data/selinawisco/kochild/five_fold_datasets/test_fold_0.csv"}, delimiter=",")["test"]


#remove unnecessary columns
train = train.remove_columns(['disease_type', 'age', 'gender','subgroup', 'id', 'textgrid_text', 'target_text','asr_text', 'new_label'])
test = test.remove_columns(['disease_type', 'age', 'gender','subgroup', 'id', 'textgrid_text', 'target_text','asr_text', 'new_label'])
train = train.cast_column("audio", Audio(sampling_rate=16_000))
test = test.cast_column("audio", Audio(sampling_rate=16_000))

print(train)
print(test)



##############################################################################   
# Finetuning setup --------------------------------------------------------- #    

# 모델 정의 ----------------------------------------
from transformers import Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer("./vocab_jrnl.json", unk_token='[UNK]', pad_token = "[PAD]", word_delimeter_token="|")

from transformers import Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# 데이터셋 프리 프로세싱 ------------------------------
def prepare_dataset(batch):
     
    audio = batch['audio']
    
    batch['input_values'] = processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]
    
    with processor.as_target_processor():
        batch['labels'] = processor(batch['human_text_jamo']).input_ids
    return batch

train = train.map(prepare_dataset, remove_columns=train.column_names,num_proc=1)
test = test.map(prepare_dataset, remove_columns=test.column_names,num_proc=1)


# DATA COLLATOR 패딩 함수 정의 ------------------------
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    
    processor: Wav2Vec2Processor
    padding: Union[bool,str]= True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.pad(
            input_features,
            padding = self.padding,
            max_length = self.max_length,
            pad_to_multiple_of = self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding = self.padding,
                max_length = self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # replace padding with -100 to ignore loss correctly

        batch["labels"] = labels
        
        # batch["input_values"] = batch["input_values"].to(device)
        # batch["labels"] = labels.to(device)
        
        return batch
    
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


# 매트릭 정의--------------------------------------------
import numpy as np

# wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids)
    
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # for i in range(3):
    #     print(f'Prediction: {pred_str[i]}')
    #     print(f"Label: {label_str[i]}")
    
    # wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = jiwer.cer(pred_str, label_str)
    return {"cer": cer}


# 모델 생성 ---------------------------------------
from transformers import Wav2Vec2ForCTC

for seed in range(42, 44):  # Modify this range to run the script 5 times with incrementing seeds
    print(f"Running training with seed {seed}")

    # Set the new seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
        
            
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-xls-r-300m", 
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )


    # model.to(device)
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    from transformers import TrainingArguments

    # 트레인 Argument 정의 ---------------------------

    training_args = TrainingArguments(
        seed=seed,
        output_dir=f"/data/selinawisco/temp/finetuning_baseline_{seed}_1",
        group_by_length=True,
        # per_device_train_batch_size=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False, #True,
        # gradient_checkpointing_kwargs={'use_reentrant':True},
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        save_steps=100,
        eval_steps=1000,
        logging_steps=10,
        learning_rate = 3e-4,
        warmup_steps=500,
        save_total_limit=2,
        # push_to_hub = True,
    )

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    tokenizer.save_pretrained(training_args.output_dir) 
    feature_extractor.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    trainer.evaluate()

