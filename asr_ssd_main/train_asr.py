from transformers import Trainer
from CustomTrainer import CustomTrainer
from transformers import TrainingArguments
import torch
import warnings
import os

def train_asr(model, data_collator, processor, tokenizer,feature_extractor, train, test, compute_metrics, config, curr_seed):
    if config.loss_feature is None:
        config.loss_feature = 'baseline'
    training_args = TrainingArguments(
            seed=curr_seed,
            # output_dir=f"/data/selinawisco/spring/wav2vec2-word-level-asr_{config.train_filename}_{config.loss_feature}_local_attention_only_{curr_seed}", # 모델 저장 경로 
            # output_dir=f"/data/selinawisco/spring/0508_{config.loss_feature}_embedding_{curr_seed}", # 모델 저장 경로 
            output_dir=f"/data/selinawisco/wav2vec2-word-level-asr-baseline_{config.train_filename}_{curr_seed}",
            group_by_length=True,
            # per_device_train_batch_size=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            gradient_checkpointing=False, #True,
            evaluation_strategy="steps",
            num_train_epochs=config.epochs,
            fp16=True,
            save_steps=100,
            eval_steps=1000,
            # eval_steps=20,
            logging_steps=10,
            learning_rate = 3e-4,
            warmup_steps=500,
            save_total_limit=2,
            # push_to_hub = True,
        )
    
    print(f'saving model in {training_args.output_dir=}')

    trainer = CustomTrainer(
        config=config,
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=processor.feature_extractor,
    )

    torch.cuda.empty_cache()
    trainer.train()
    # trainer.train(resume_from_checkpoint='/data/selinawisco/wav2vec2-word-level-asr-baseline_five_fold_datasets/test_fold_0_train.csv_42')
    
    tokenizer.save_pretrained(training_args.output_dir) 
    feature_extractor.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    
    # temporary fix : unknow error in evaluate
    try:
        trainer.evaluate(model)
    except Exception as e:
        warnings.warn("evaluation loop crashed, skipping . . .")
        return None
        
        
    # test(model, training_args.output_dir, processor, curr_seed)
    

