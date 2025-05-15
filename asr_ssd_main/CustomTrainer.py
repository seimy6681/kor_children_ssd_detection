import pandas as pd
import debugpy
import jiwer
# import torchaudio
import random
from transformers import Trainer, get_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm
import wandb
from typing import Any, Tuple, Union, List, Callable, Optional
from features.target_text_dict import target_text_dict

# HARD CODING WARNING: IF USING LOSS_FEATURE AS SELECT_DIFFICULT_WORDS, PLEASE PRE DEFINE THE SELECTED WORD LIST HERE BEFORE TRAINING:
SELECTED_DIFFICULT_WORDS = ['색종이', '화장실', '머리', '호랑이','컵','사탕']
SELECTED_DIFFICULT_WORDS = [target_text_dict.get(word, None) for word in SELECTED_DIFFICULT_WORDS]

class CustomTrainer(Trainer):
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config  # Save config as an instance attribute
     
    # def create_optimizer_and_scheduler(self, num_training_steps):
    #     """
    #     Override optimizer to use different learning rates for ASR and phoneme classifiers.
    #     """
    #     if self.optimizer is None:
    #         # Separate learning rates
    #         asr_params = list(self.model.wav2vec2.parameters())  # ASR base model
    #         phoneme_params = list(self.model.error_detection_head.parameters())  # Phoneme classifier

    #         optimizer_grouped_parameters = [
    #             {"params": asr_params, "lr": 3e-4},  # ASR learning rate
    #             {"params": phoneme_params, "lr": 1e-4},  # Lower LR for phoneme classifier
    #         ]

    #         # AdamW optimizer with weight decay
    #         self.optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    #     # self.lr_scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs
        
    #     num_training_steps = len(self.config.train_dataloader) * self.config.num_epochs

    #     # Linear warmup + linear decay scheduler
    #     self.lr_scheduler = get_scheduler(
    #         name="linear",
    #         optimizer=self.optimizer,
    #         num_warmup_steps=500,  # Default in HF fine-tuning
    #         num_training_steps=num_training_steps
    #     )

    def compute_loss(self, model, inputs, return_outputs=False):

        if self.config.loss_feature == 'word_cer':
            loss_feature = inputs.pop(self.config.loss_feature)
            outputs = model(**inputs)
            base_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            weights = 1 + loss_feature.to(base_loss.device) # loss 크게
            weighted_loss = base_loss * weights.view(-1)
            
            return (weighted_loss.mean(), outputs) if return_outputs else weighted_loss.mean()
    
        if self.config.loss_feature == 'target_text_id':
            loss_feature = inputs.pop(self.config.loss_feature)
            # check if the current target word is in the custom difficult level word list
            outputs = model(**inputs)

            # logits = outputs.logits
            # labels = inputs["labels"]
            # log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # ctc_loss_fn = nn.CTCLoss(blank=model.config.pad_token_id, reduction='none',zero_infinity=True)
            # input_lengths = torch.full(size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long)
            # label_lengths = torch.full(size=(labels.size(0),), fill_value=labels.size(1), dtype=torch.long)
            
            # losses = ctc_loss_fn(log_probs.transpose(0,1), labels, input_lengths, label_lengths)
            
            base_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            difficult_words = torch.tensor(SELECTED_DIFFICULT_WORDS, device=base_loss.device) # Convert diff.word list into tensor for comparsion
            mask = torch.isin(loss_feature, difficult_words)
            
            epsilon = 10**(-8)
            dynamic_scaling_factor = 1 + (base_loss / (base_loss + epsilon)) * 1.5
            
            weighted_loss = torch.where(mask, base_loss*1.5, base_loss)
            # weighted_loss = base_loss ** 2 # square the loss for the difficult level word
    
        if self.config.loss_feature == 'phoneme_error_label' or self.config.loss_feature == 'binary_classifier_by_phoneme':
            outputs = model(**inputs)
            # Extract phoneme classification loss separately (if available)
            phoneme_error_loss = outputs.phoneme_error_loss if hasattr(outputs, "phoneme_error_loss") else None

            # Log losses to Weights & Biases
            log_data = {}  # Log main loss
            log_data["train/Multitask loss (CTC + PE loss)"] = outputs.loss if hasattr(outputs, "loss") else None
            if phoneme_error_loss is not None:
                log_data["train/Phoneme Error loss"] = phoneme_error_loss.mean().item()  # Log phoneme loss

            wandb.log(log_data)  # Send logs to Weights & Biases
            weighted_loss = outputs["loss"] if isinstance(outputs,dict) else outputs[0]
            return (weighted_loss.mean(), outputs) if return_outputs else weighted_loss.mean()

        if self.config.adversarial:
            # # implementing reverse layer alpha scheduler
            # global_step = self.state.global_step
            # max_step = self.state.max_steps
            # p = global_step / max_step
            # max_val = 1.0
            # gamma = 10.0
            # alpha_reverse = max_val * (2. / (1. + np.exp(-gamma * p )) -1 )
            
            # inputs["reverse_alpha"] = alpha_reverse # inputs to forward()
            
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return (loss.mean(),outputs) if return_outputs else loss.mean()
            
        else:
            
            # loss_feature = inputs.pop(self.config.loss_feature)
            outputs = model(**inputs)
            weighted_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # weighted_loss = base_loss # trivial
            
        return (weighted_loss.mean(), outputs) if return_outputs else weighted_loss.mean()
            