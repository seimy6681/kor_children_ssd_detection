# WhisperForConditionalGeneration 대신  models/ 안에 multitask_whisper_for_conditional_generation.py 모델을 불러와 asr finetuning 한다.
# https://huggingface.co/blog/fine-tune-whisper

import torch
import os
from datasets import load_dataset
from datasets import Audio
import debugpy
import wandb
from transformers import GenerationConfig
from transformers import EvalPrediction

from multitask_trainer import MultitaskTrainer
import argparse
from argparse import Namespace, ArgumentParser
from typing import Any, Dict
import json
from transformers import WhisperConfig, Trainer
from util.extract_vocab import extract_vocab

# os.environ["HF_DATASETS_CACHE"] = "/data/selinawisco/hf_cache"

# debugpy.listen(5678)
# print("waiting for debugger")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('break on this line')

class Parser(object):
    def __call__(self, parser: ArgumentParser, args: Namespace) -> Dict[str, Dict[str, Any]]:
        config = dict()
        for group in parser._action_groups:
            group_dict={a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            for k, v in group_dict.copy().items():
                if v == None:
                    group_dict.pop(k, None)
            if len(group_dict) > 0:
                config[group.title] = group_dict
                
        return config
    
num_runs = 1 # 연속으로 학습할 수
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
default_group = parser.add_argument_group('default')
default_group.add_argument('--train_file', type=str, help="file name of training data")
default_group.add_argument('--test_file', type=str, help="file name of testing data")
default_group.add_argument('--fold', type=str, help="train-test set fold (0-4)")

if __name__=='__main__':
    arg = parser.parse_args()
    arg_parser = Parser()
    args = arg_parser(parser, arg)

    # CSV 경로 설정 --------------------------------------------------------------------------
    DATA_PATH = '/data/selinawisco/kochild/wav2vec2-fold-0-phoneme_segments/'
    train = load_dataset("csv", data_files={"train": DATA_PATH +  "fold_0_train_phoneme_segment.csv"}, delimiter=",")["train"]
    # train = load_dataset("csv", data_files={"train": DATA_PATH + config.train_file}, delimiter=",")["train"]
    # train = load_dataset("csv", data_files={"train": DATA_PATH + "test_fold_1_train.csv"}, delimiter=",")["train"]
    # train = load_dataset("csv", data_files={"train": '/home/selinawisco/selina_main/asr/01_asr_train_sample.csv'}, delimiter=",")["train"]
    test = load_dataset("csv", data_files={"test": DATA_PATH + "fold_0_test_phoneme_segment.csv"}, delimiter=",")["test"]
    # test = load_dataset("csv", data_files={"test": DATA_PATH + "test_fold_1.csv"}, delimiter=",")["test"]
    # ---------------------------------------------------------------------------------------

    train = train.filter(lambda entry: entry['phoneme'] not in ['[', ']','?','ㅣ', ' ', ''])
    test = test.filter(lambda entry: entry['phoneme'] not in ['[', ']','?','ㅣ', ' ', ''])
    
    train = train.rename_column('phoneme', 'label')
    test = test.rename_column('phoneme', 'label')
    train = train.rename_column('segment_audio', 'audio')
    test = test.rename_column('segment_audio', 'audio')

    # train = train.remove_columns(['disease_type', 'age', 'gender','subgroup', 'id', 'textgrid_text', 'target_text','asr_text','target_text_jamo','human_text'])
    # test = test.remove_columns(['disease_type', 'age', 'gender','subgroup', 'id', 'textgrid_text', 'target_text','asr_text','target_text_jamo','human_text'])

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

    vocab = extract_vocab(train,test)
    phoneme_to_id = {p:i for i,p in enumerate(vocab)}

    print(phoneme_to_id)
    
    def prepare_dataset(batch):

        audio = batch['audio']
        features = feature_extractor(
            audio['array'], 
            sampling_rate=audio['sampling_rate'],
            # return_attention_mask=True, # add attention_mask so it can be accessed for binary classifier,
            # padding="max_length",
            # truncation=True,
            # max_length=600,
        )
        
        batch['input_features']= features.input_features[0]
        # batch['attention_mask']= features.attention_mask[0]
        batch["labels"] = phoneme_to_id.get(batch["label"])
        
        return batch

    train = train.map(prepare_dataset, num_proc=1)
    test = test.map(prepare_dataset, num_proc=1)

    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    @dataclass
    class DataCollatorForPhonemeClassification:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                # Extract the raw input_features list for padding
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # Add attention mask to the batch
            # if "attention_mask" in features[0]:
            #     attention_masks = [torch.tensor(f["attention_mask"], dtype=torch.float) for f in features]
            #     batch["attention_mask"] = torch.stack(attention_masks).float()

            if "labels" in features[0]:
                batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)

            return batch

    data_collator = DataCollatorForPhonemeClassification(processor=processor)
    
    print(test[0].keys())
    # from transformers import WhisperForConditionalGeneration
    from whisper_models.whisper_for_phoneme_classification import WhisperForPhonemeClassification
    config = WhisperConfig.from_pretrained("openai/whisper-small")
    config.num_labels = len(vocab)
    
    # # add phoneme id mappings
    config.label2id = phoneme_to_id
    config.id2label = {v: k for k,v in phoneme_to_id.items()}
    
    print("number of phonemes in train-test: ",config.num_labels)

    model = WhisperForPhonemeClassification.from_pretrained('openai/whisper-small', config=config)
    # model = torch.nn.DataParallel(model)

    import evaluate
    import numpy as np
    from sklearn.metrics import recall_score,accuracy_score
    metric = evaluate.load("cer")

    def compute_metrics(pred):

        logits = pred.predictions # get phoneme classification logits
        # pred_ids = pred.predictions
        labels = pred.label_ids # get both asr logits and binary classification logits
        
        preds = np.argmax(logits, axis=-1)
        uar = recall_score(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        
        return {"uar": uar, "acc" :acc}


    from transformers import TrainingArguments,Seq2SeqTrainingArguments


    for seed in range(42, 42 + 1): 
        print(f"Running training with seed {seed}, dataset fold {arg.fold}")

        # Set the new seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
            
        wandb.init(
        project="whisper-finetuning",
        name=f"phoneme-classification-whisper-fold-{arg.fold}-seed-{seed}",
        config={
            "learning_rate": 3e-4,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "fp16": True,
            "max_steps": 5000,
            "seed": seed,
            "alpha" : 0.3,
            "train_file": arg.train_file,
            "test_file":arg.test_file
        },
    )
            
        # training_args = TrainingArguments(
        #     seed=seed,
        #     output_dir=f"/data/selinawisco/whisper_finetuning_asr/phoneme-classification-whisper-small-fold{arg.fold}-0.15-{seed}",  # change to a repo name of your choice
        #     group_by_length = True,
        #     per_device_train_batch_size = 16,
        #     per_device_eval_batch_size = 16,
        #     dataloader_num_workers=1,
        #     dataloader_pin_memory=True,
        #     evaluation_strategy = "epoch",
        #     save_strategy = "epoch",
        #     logging_strategy = "epoch",
        #     save_total_limit = 2,
        #     num_train_epochs = 10,
        #     # num_train_epochs =args.num_epochs,
        #     fp16 = True,
        #     learning_rate = 3e-4,
        #     # learning_rate = args.learning_rate,
        #     gradient_checkpointing = False, 
        #     gradient_accumulation_steps = 1,
        #     weight_decay = 0.01,
        #     # weight_decay = args.weight_decay,
        #     push_to_hub = False,  
        #     report_to = "wandb"
        # )

        training_args = Seq2SeqTrainingArguments(
            seed=seed,
            output_dir=f"/data/selinawisco/wav2vec2-fold-0-phoneme-classification-whisper-small-fold-{seed}",  # change to a repo name of your choice
            save_total_limit=2,
            # per_device_train_batch_size=16,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
            # gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=3e-4,
            warmup_steps=500,
            max_steps=10000,
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
            load_best_model_at_end=False,
            # load_best_model_at_end=True,
            metric_for_best_model="uar",
            greater_is_better=False,
            push_to_hub=False,
            remove_unused_columns=False,
            report_to=["wandb"]
        )


        trainer = Trainer(
            args=training_args,
            model=model,
            train_dataset=train,
            eval_dataset=test,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )

        # saving the phoneme to id dictionary in the output_directory for evaluation use
        with open(f"{training_args.output_dir}/phoneme_to_id.json", "w") as f:
            json.dump(phoneme_to_id, f, ensure_ascii=False)
        
        # trainer.train(resume_from_checkpoint="/data/selinawisco/whisper_finetuning_asr/wav2vec2-fold-0-phoneme-classification-whisper-small-foldNone-0.15-42/checkpoint-5000")
        config.save_pretrained(training_args.output_dir)
        trainer.train()
        tokenizer.save_pretrained(training_args.output_dir) 
        feature_extractor.save_pretrained(training_args.output_dir)
        model.save_pretrained(training_args.output_dir)
        
        # trainer.evaluate()
        
    wandb.finish()


            