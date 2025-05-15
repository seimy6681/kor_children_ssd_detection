import torch
from torch import nn
from transformers import Wav2Vec2ForCTC
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import CausalLMOutput
from dataclasses import dataclass


@dataclass
class MultitaskWav2Vec2Output(CausalLMOutput):
    """
    Custom output for multitask Wav2Vec2 with both ASR logits and binary classification logits.
    """
    error_logits: Optional[torch.FloatTensor] = None  # Binary classification logits


class DirectAttentionMultitaskWav2Vec2ForCTC(Wav2Vec2ForCTC):
    """
    Wav2Vec2ForCTC modified to replace original hidden states with attention-weighted hidden states
    for error detection and multitask learning.
    """

    def __init__(self, config):
        super().__init__(config)
        
        # Binary classification head for error detection
        self.error_detection_head = nn.Linear(config.hidden_size, 2)  # Output logits for binary classification
        self.alpha = getattr(config, "multitask_alpha", 0.15)

        # Attention layer for focusing on key parts of the sequence
        self.attention_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1),
            nn.Softmax(dim=1),
        )

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
        Forward pass with replaced attention mechanism.
        """
        # Forward pass through the base Wav2Vec2 model
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=return_dict
        )

        # Extract the last hidden states
        hidden_states = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        hidden_states = self.dropout(hidden_states)

        # Attention mechanism to compute attended states
        attention_scores = self.attention_layer(hidden_states)  # Shape: (batch_size, seq_len, 1)
        attention_weights = attention_scores.squeeze(-1)  # Shape: (batch_size, seq_len)
        attended_states = torch.bmm(
            attention_weights.unsqueeze(1),  # Shape: (batch_size, 1, seq_len)
            hidden_states  # Shape: (batch_size, seq_len, hidden_size)
        ).squeeze(1)  # Shape: (batch_size, hidden_size)

       

        # ASR Logits (using the replaced hidden states)
        logits = self.lm_head(attended_states)  # Shape: (batch_size, seq_len, vocab_size)

        # Binary classification logits for error detection
        error_logits = self.error_detection_head(attended_states)  # Shape: (batch_size, 2)

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
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
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

        return MultitaskWav2Vec2Output(
            loss=loss,
            logits=logits,
            error_logits=error_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
