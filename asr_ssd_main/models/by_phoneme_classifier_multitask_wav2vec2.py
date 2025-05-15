import torch
from torch import nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC
from typing import Optional, Union ,Tuple
from transformers.modeling_outputs import CausalLMOutput
# from transformers.modeling_outputs import ModelOutput
# from .multitask_wav2vec2_output import MultitaskWav2Vec2Output
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_HIDDEN_STATES_START_POSITION = 2
PHONEME_ERROR_LABEL_MAX_LEN = 11

@dataclass
class MultitaskWav2Vec2Output(CausalLMOutput):
    """
    Custom output class for multitask Wav2Vec2 that includes:
    - `logits`: ASR logits
    - `phoneme_error_logits`: Binary classification logits for phoneme correctness
    """ 
    phoneme_error_logits: Optional[torch.FloatTensor] = None  # Shape: (batch_size, max_target_length, 2)
    phoneme_error_loss: Optional[torch.FloatTensor] = None  # Binary classification loss


class ByPhonemeClassifierMultitaskWav2Vec2ForCTC(Wav2Vec2ForCTC):
    """
    Wav2Vec2ForCTC extended to include an auxiliary binary classification(correct/incorrect pronunciation) head for multitask learning.
    """

    def __init__(self,config):

        super().__init__(config)
        print(config)
        # Binary Classification Head: A fully connected (linear) layer that outputs 2 logits for binary classification (e.g., error vs. no error).
        self.error_detection_head = nn.Linear(config.hidden_size, 2) # [1, 0] if target==human (correct), [0, 1] if target!=human (incorrect)
        self.error_detection_heads = nn.ModuleDict({
            str(i): nn.Linear(config.hidden_size,2) for i in range(46)
        })

        self.alpha = getattr(config, "multitask_alpha", 0.3)
     
    #This function groups the logits(hidden states) by phoneme segments, 
    # averages the logits per segment, 
    # and aligns them with one hot target labels for binary classification loss computation per phoneme segment. 
    def aggregate_phoneme_logits(self, logits, pred_tokens, input_lengths):
        
        # logits = logits.permute(1,0,2)
        batch_size, seq_len, vocab_size = logits.shape
        batch_logits = []
        
        for i in range(batch_size): 
            
            input_len = input_lengths[i] # only process this many time steps for the prediction logits
                        
            curr_word_logits = []
            prev_token = None
            segment_logits = []
            
            token_sequence = pred_tokens[:,i] # shape == seq_len
            # print(f'{i}/{batch_size}-th token sequence: ')
            # print(f'{token_sequence=}')
            
            for t in range(input_len): # iterating over time steps in the token sequence
                
                cur_token = token_sequence[t].item() #  t-th token in the current token sequence prediction
                # print(f'{t}/{input_len}-th token is \'{cur_token}\'')
                
                if cur_token == 45:
                    continue
                
                last_logit = (t == input_len -1) # is it the last logit ?
                
                if cur_token == prev_token or prev_token is None: # if phoneme doesn't change, keep collecting logits

                    segment_logits.append(logits[i, t]) # shape == vocab_size (46)

                    
                if prev_token is not None and (cur_token != prev_token or last_logit) and segment_logits: # if phoneme changed or it's the last time step, finalize the segment

                    avg_logits = torch.stack(segment_logits).mean(dim=0) # taking the average of all the logits collected for the current phoneme segment
                    
                    curr_word_logits.append(avg_logits) # adding the final avg. phoneme logit

                    segment_logits = [logits[i,t]] # reset for next phoneme segment
                    
                prev_token = cur_token
                
            # pad or trunclated to match target_length
            pad_size = PHONEME_ERROR_LABEL_MAX_LEN -  len(curr_word_logits)

            if pad_size > 0:
                curr_word_logits.extend([torch.zeros(vocab_size, device=logits.device)] * pad_size)
            elif pad_size < 0:
                curr_word_logits = curr_word_logits[:PHONEME_ERROR_LABEL_MAX_LEN] # cut down to 11 by 46
            curr_word_logits_tensor = torch.stack(curr_word_logits, dim=0) # list of 11 (46,) tensors -> (11, 46) tensor    

            batch_logits.append(curr_word_logits_tensor)
            # truncate the pred sequence if it's longer than the target len
        
            
        batch_logits = torch.stack(batch_logits) # (8, 11, 46)
            
            # segment_labels.append(one_hot_targets[i, :len(batch_logits)].to(logits.device))
        
        return batch_logits

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        target_phonemes: Optional[torch.Tensor] = None,
        phoneme_error_label: Optional[torch.Tensor] = None,  # phoneme level binary label
    ) -> Union[Tuple, CausalLMOutput]:
        
        """
        Forward pass with auxiliary error detection.
        """

        # Forward pass through the base Wav2Vec2 model
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=False, ###
            return_dict=return_dict

        )

        # Extract the last hidden states
        hidden_states = outputs.last_hidden_state # shape: (batch_size, seq_len, hidden_size)
        hidden_states = self.dropout(hidden_states)

        # the ASR Logits
        logits = self.lm_head(hidden_states)

        loss = None
        loss_asr = None
        loss_phoneme = None

        # CTC Loss for ASR
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0,1)
        
        # pred_tokens = torch.argmax(log_probs, dim=-1)
        top2_tokens = log_probs.topk(2, dim=-1).indices  # Get top 2 token indices per timestep, [154,8,2]
        pred_tokens = torch.where(top2_tokens[:, :, 0] == 45, top2_tokens[:, :, 1], top2_tokens[:, :, 0]) # [154,8]

        
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        target_lengths = None
        
        if labels is not None:
            
            
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            with torch.backends.cudnn.flags(enabled=False):
                loss_asr = nn.functional.ctc_loss(
                    log_probs, # Number of valid time steps after feature extraction.
                    flattened_targets, # Ground truth phoneme sequence ((valid_phonemes,))
                    input_lengths, # Number of valid time steps after feature extraction.
                    target_lengths, # Number of ground truth phonemes per batch sample
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity, # Prevents NaN loss values when target is longer than input

                )
            # print(f'{input_lengths=}')
            # print(f'{target_lengths=}')
            
        # if phoneme_error_label is not None:

        batch_phoneme_logits = self.aggregate_phoneme_logits(hidden_states, pred_tokens, input_lengths) # (8, 11, 1024)
                
    
        if target_lengths is not None:
            # Training: Use target_lengths to create mask
            valid_mask = torch.arange(PHONEME_ERROR_LABEL_MAX_LEN, device=logits.device).unsqueeze(0) < target_lengths.unsqueeze(1)
        else:
            # Inference: Use input_lengths to determine valid phoneme regions
            valid_mask = torch.arange(PHONEME_ERROR_LABEL_MAX_LEN, device=logits.device).unsqueeze(0) < input_lengths.unsqueeze(1)

       
        if target_phonemes is not None:
            
            
            # # vectorized version !!!!!!!!! ################################################
            # # Flatten (B, T, H) -> (B*T, H)
            # flat_logits = batch_phoneme_logits.reshape(-1, batch_phoneme_logits.shape[-1])        # (B*T, H)
            # flat_target_phonemes = target_phonemes.reshape(-1)                                     # (B*T)
            # flat_valid_mask = valid_mask.reshape(-1).bool()                                        # (B*T)
            
            # # filter valid (non padded) entries
            # valid_logits = flat_logits[flat_valid_mask]
            # valid_phonemes = flat_target_phonemes[flat_valid_mask]
            
            
            # logits_per_sample = []
            # for logit, phoneme_id in zip(valid_logits, valid_phonemes):
            #     classifier = self.error_detection_heads[str(int(phoneme_id.item()))]
            #     logits_per_sample.append(classifier(logit))
            
            # phoneme_binary_logits = torch.stack(logits_per_sample) # (num_valid, 2)
            # ################################################################################

            # for loop version !!!!!!!!!!!!
            phoneme_binary_logits = []
            
            for b in range(batch_phoneme_logits.size(0)):  # Iterate over batch
                sample_logits = []
                for t in range(PHONEME_ERROR_LABEL_MAX_LEN):  # Iterate over phoneme segments (max 11)
                    if not valid_mask[b, t]:
                        sample_logits.append(torch.zeros(2, device=logits.device))  # Padding
                        continue

                    logit = batch_phoneme_logits[b, t]  # (hidden_size)
                    phoneme_id = int(target_phonemes[b, t].item())

                    # Apply the correct phoneme-specific classifier
                    if str(phoneme_id) in self.error_detection_heads:
                        classifier = self.error_detection_heads[str(phoneme_id)] # 1024 -> 2
                        pred = classifier(logit)  # Output shape: (2)
                    else:
                        pred = torch.zeros(2, device=logits.device)

                    sample_logits.append(pred)

                sample_logits = torch.stack(sample_logits, dim=0)  # (11, 2)
                phoneme_binary_logits.append(sample_logits)

            phoneme_binary_logits = torch.stack(phoneme_binary_logits, dim=0)  # (B, 11, 2)


        if phoneme_error_label is not None:
            
            
            padded_segment_labels = []
            for label_seq in phoneme_error_label:
                padded = []
                for label in label_seq:
                    if label == -100:
                        padded.append(torch.tensor([0.0, 0.0], device=label_seq.device))  # padding
                    else:
                        one_hot = F.one_hot(label.to(torch.int64), num_classes=2).float()
                        padded.append(one_hot)
                # pad to length 11 if needed
                while len(padded) < PHONEME_ERROR_LABEL_MAX_LEN:
                    padded.append(torch.tensor([0.0, 0.0], device=label_seq.device))
                padded_segment_labels.append(torch.stack(padded[:PHONEME_ERROR_LABEL_MAX_LEN]))

            segment_labels = torch.stack(padded_segment_labels)  # shape: (B, 11, 2)
            
            # loss_fn = nn.BCEWithLogitsLoss(reduction='none') # per token loss
            class_weights = torch.tensor([0.3, 0.7], device=phoneme_binary_logits.device)  # Adjust based on your class distribution
            # loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            loss_fn = nn.BCEWithLogitsLoss(weight=class_weights,reduction='none')

            loss_phoneme = loss_fn(phoneme_binary_logits, segment_labels)
            loss_phoneme = loss_phoneme * valid_mask.unsqueeze(-1).float()
            loss_phoneme = loss_phoneme.sum(dim=-1)  # sum over class dimension → (B, T)
            per_sample_loss = loss_phoneme.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)  # (B,)

            loss_phoneme = per_sample_loss.mean()  # scalar
                        
            # loss_phoneme = loss_phoneme.mean()

            phoneme_binary_logits = phoneme_binary_logits * valid_mask.unsqueeze(-1).float().detach()

        if loss_asr is not None or loss_phoneme is not None:
            loss = (loss_asr if loss_asr is not None else 0) + self.alpha * (loss_phoneme if loss_phoneme is not None else 0)

        if not return_dict:
            return (loss, logits, phoneme_binary_logits) + outputs.hidden_states

        return MultitaskWav2Vec2Output(
            loss=loss,
            logits=logits, # asr_logits
            phoneme_error_logits=phoneme_binary_logits,
            phoneme_error_loss=loss_phoneme,  
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )