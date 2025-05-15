import torch
from torch import nn
from transformers import Wav2Vec2ForCTC
from typing import Optional, Union ,Tuple
from transformers.modeling_outputs import CausalLMOutput
# from transformers.modeling_outputs import ModelOutput
# from .multitask_wav2vec2_output import MultitaskWav2Vec2Output
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

_HIDDEN_STATES_START_POSITION = 2


@dataclass
class MultitaskWav2Vec2Output(CausalLMOutput):
    """
    Custom output for multitask Wav2Vec2 with both ASR logits and binary classification logits.
    """
    error_logits: Optional[torch.FloatTensor] = None  # Binary classification logits


class BinaryMultitaskWav2Vec2ForCTC(Wav2Vec2ForCTC):
    """
    Wav2Vec2ForCTC extended to include an auxiliary binary classification(correct/incorrect pronunciation) head for multitask learning.
    """

    def __init__(self,config):

        super().__init__(config)
        print(config)
        # Binary Classification Head: A fully connected (linear) layer that outputs 2 logits for binary classification (e.g., error vs. no error).
        self.error_detection_head = nn.Linear(config.hidden_size, 2) # [1, 0] if target==human (correct), [0, 1] if target!=human (incorrect)
        self.alpha = getattr(config, "multitask_alpha", 0.15)
        
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        binary_label: Optional[torch.Tensor] = None,  # New for auxiliary task
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

        # Error detection logits (mean pooling over sequence)
        pooled_hidden_states = hidden_states.mean(dim=1) # Shape: (batch_size, hidden_size)
        error_logits = self.error_detection_head(pooled_hidden_states) # shape: (batch_size, 2)

        loss = None
        loss_asr = None
        loss_error = None

        if labels is not None:
            
            attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # CTC Loss for ASR
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0,1)
            with torch.backends.cudnn.flags(enabled=False):
                loss_asr = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,

                )

        if binary_label is not None:
            # Binary cross entropy loss for error detection
            loss_fn = nn.BCEWithLogitsLoss()   
            loss_error = loss_fn(error_logits, binary_label)

        if loss_asr is not None or loss_error is not None:
            loss = (loss_asr if loss_asr is not None else 0) + self.alpha * (loss_error if loss_error is not None else 0)

        if not return_dict:
            return (loss, logits, error_logits) + outputs.hidden_states
        
        # return CausalLMOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

        return MultitaskWav2Vec2Output(
            loss=loss,
            logits=logits,
            error_logits=error_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )