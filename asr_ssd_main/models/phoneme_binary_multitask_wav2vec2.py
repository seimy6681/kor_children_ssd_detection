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


class PhonemeBinaryMultitaskWav2Vec2ForCTC(Wav2Vec2ForCTC):
    """
    Wav2Vec2ForCTC extended to include an auxiliary binary classification(correct/incorrect pronunciation) head for multitask learning.
    """

    def __init__(self,config):

        super().__init__(config)
        print(config)
        # Binary Classification Head: A fully connected (linear) layer that outputs 2 logits for binary classification (e.g., error vs. no error).
        self.error_detection_head = nn.Linear(config.hidden_size, 2) # [1, 0] if target==human (correct), [0, 1] if target!=human (incorrect)
        # self.error_detection_head = nn.Linear(46, 2) # [1, 0] if target==human (correct), [0, 1] if target!=human (incorrect)
        self.alpha = getattr(config, "multitask_alpha", 0.3)
        
        # Create phoneme-specific classification heads
        # self.error_detection_heads = nn.ModuleDict({
        #     str(i): nn.Linear(config.hidden_size, 2) for i in range(46)
        # })  
     
    #This function groups the logits(hidden states) by phoneme segments, 
    # averages the logits per segment, 
    # and aligns them with one hot target labels for binary classification loss computation per phoneme segment. 
    def aggregate_phoneme_logits(self, logits, pred_tokens, input_lengths):
        
        # logits = logits.permute(1,0,2)
        batch_size, seq_len, vocab_size = logits.shape
        # aggregated_logits = [] # average logits for each phoneme
        # segment_labels = [] # corresponding phoneme labels
         
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
                    # if prev_token in target_tokens[i]:
                    avg_logits = torch.stack(segment_logits).mean(dim=0) # taking the average of all the logits collected for the current phoneme segment
                    # print(f'average prob distribution for {t}/{input_len}-th token {cur_token} is -> {avg_logits=}')
                    
                    curr_word_logits.append(avg_logits) # adding the final avg. phoneme logit
                    # print(f'logit for pred_token {prev_token} added')
                    # print('i-th token_sequence in the batch is now: ')
                    # print(f'{curr_word_logits=}')
                    segment_logits = [logits[i,t]] # reset for next phoneme segment
                    
                prev_token = cur_token
                
            # pad or trunclated to match target_length
            pad_size = PHONEME_ERROR_LABEL_MAX_LEN -  len(curr_word_logits)
            # print(f'{i}-th sample segmentation length : {len(curr_word_logits)}')

            if pad_size > 0:
                curr_word_logits.extend([torch.zeros(vocab_size, device=logits.device)] * pad_size)
            elif pad_size < 0:
                curr_word_logits = curr_word_logits[:PHONEME_ERROR_LABEL_MAX_LEN] # cut down to 11 by 46
            curr_word_logits_tensor = torch.stack(curr_word_logits, dim=0) # list of 11 (46,) tensors -> (11, 46) tensor    
            # print(f'{i}-th word logits sequence in the batch added to batch_logits, ')
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

        
        # Apply binary classification head after stacking (batch_size, max_target_length, 2)
        phoneme_binary_logits = self.error_detection_head(batch_phoneme_logits) # (8, 11, 2)
        
        # phoneme_binary_logits = []
        # for batch in batch_phoneme_logits:
        #     phoneme_logits = []
        #     for phoneme, logit in batch:
        #         if str(phoneme) in self.error_detection_heads:
        #             phoneme_logits.append(self.error_detection_heads[str(phoneme)](logit))
        #         else:
        #             phoneme_logits.append(torch.zeros(2, device=logits.device))
        #     phoneme_binary_logits.append(torch.stack(phoneme_logits))

        # phoneme_binary_logits = torch.stack(phoneme_binary_logits)

        if phoneme_error_label is not None:

            phoneme_error_label = phoneme_error_label.clone()
            phoneme_error_label[phoneme_error_label == -100] = 0  # Safe placeholder

            # loss_fn = nn.BCEWithLogitsLoss(reduction='none') # per token loss
            class_weights = torch.tensor([0.3, 0.7], device=phoneme_binary_logits.device)  # Adjust based on your class distribution
            loss_fn = nn.BCEWithLogitsLoss(weight=class_weights)

            # loss_fn = nn.BCEWithLogitsLoss(weight=valid_mask.unsqueeze(-1).float(), reduction='none')

            segment_labels_one_hot = F.one_hot(phoneme_error_label.to(torch.int64), num_classes=2).float() # if phoneme_error_label is [0 0 1 1] -> [1 0], [1 0], [0 1], [0 1]
            #Pad `segment_labels` to match `max_target_length`
            segment_labels = [
                F.pad(label, (0, 0, 0, PHONEME_ERROR_LABEL_MAX_LEN - label.shape[0])) 
                if label.shape[0] < PHONEME_ERROR_LABEL_MAX_LEN else label[:PHONEME_ERROR_LABEL_MAX_LEN]
                for label in segment_labels_one_hot     
            ] # pads with [0, 0] to match 11
            segment_labels = torch.stack(segment_labels)

            loss_phoneme = loss_fn(phoneme_binary_logits, segment_labels)
            loss_phoneme = loss_phoneme * valid_mask.unsqueeze(-1).float()  # Mask shape: (batch_size, max_target_length, 2)
            loss_phoneme = loss_phoneme.sum() / (valid_mask.sum() + 1e-6)

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