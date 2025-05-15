import torch
from torch import nn
from transformers import Wav2Vec2ForCTC
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import CausalLMOutput
from dataclasses import dataclass

# The second attention layer ensures that the model focuses on key time steps 
# in the audio when deciding whether an error is present. 
# It acts like a "spotlight" that highlights the most relevant parts of the sequence. 
# By fusing this spotlight back with the full sequence, 
# the model improves its overall understanding for both tasks.

@dataclass
class MultitaskWav2Vec2Output(CausalLMOutput):
    """
    Custom output for multitask Wav2Vec2 with both ASR logits and binary classification logits.
    """
    error_logits: Optional[torch.FloatTensor] = None  # Binary classification logits


class JointAttendedBinaryMultitaskWav2Vec2ForCTC(Wav2Vec2ForCTC):
    """
    Wav2Vec2ForCTC extended to include an auxiliary binary classification head for multitask learning.
    """

    def __init__(self, config, main_arg=None):
        super().__init__(config)
        print(config)
        # Second attention layer for binary classification
        self.error_detection_head = nn.Linear(config.hidden_size, 2)  # Compute attention scores
        self.alpha = main_arg.multitask_alpha if main_arg is not None else 0.3
        
        self.second_attention_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1),
            nn.Softmax(dim=1),
        )
        
        self.fusion_layer = nn.Linear(config.hidden_size * 2, config.hidden_size) 
        
        self.joint_layer = nn.Sequential(
            nn.Linear(50, 16), # vocab size, 2, 1, 1
            nn.ReLU(),
            nn.Linear(16,2),
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
        Forward pass with auxiliary error detection.
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

        # # ASR Logits
        # logits = self.lm_head(hidden_states) # Original logits for ASR

        # Error detection with second attention
        attention_scores = self.second_attention_layer(hidden_states) # Shape: (batch_size, seq_len, 1)
        attention_weights = attention_scores.squeeze(-1) # (batch_size, seq_len)
        # (1×T) (weights) with (T×D)(T×D) (hidden states).
        attended_states = torch.bmm(
            attention_weights.unsqueeze(1), # Shape: (batch_size, 1, seq_len)
            hidden_states # Shape: (batch_size, seq_len, hidden_size)
        ).squeeze(1) # Shape: (batch_size, hidden_size)
        

        fused_states = torch.cat(
            [hidden_states, attended_states.unsqueeze(1).expand(-1, hidden_states.size(1), -1)],
            dim=-1
        ) # Shape: (batch_size, seq_len, 2 * hidden_size)
        
        fused_states = self.fusion_layer(fused_states) # shape: (batch_size, seq_len, hidden_size)
        
        # Fused ASR Logits
        logits = self.lm_head(fused_states)
        
        

        pooled_hidden_states = fused_states.mean(dim=1)
        # pooled_hidden_states = hidden_states.mean(dim=1) # original logits for AC
        
        # error_logits = self.error_detection_head(pooled_hidden_states)
        error_logits = self.error_detection_head(pooled_hidden_states)
        
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
            
            ac_confidence = torch.sigmoid(error_logits).mean(dim=1)
            asr_confidence = logits.abs().mean(dim=(1,2))
            
            # confidence_sum = ac_confidence + asr_confidence + 1e-8
            # ac_weight= ac_confidence / confidence_sum
            # asr_weight = asr_confidence / confidence_sum

            # ac_weight = torch.clamp(ac_weight, max=0.4)
            # asr_weight = 1 - ac_weight

            joint_input = torch.cat([
                logits.mean(dim=1),  # Average ASR logits (B, V)
                error_logits,  # AC logits (B, 2)
                asr_confidence.unsqueeze(-1),  # ASR confidence (B, 1)
                ac_confidence.unsqueeze(-1)  # AC confidence (B, 1)
            ], dim=-1)  # (B, V + 2 + 2)

            joint_output = self.joint_layer(joint_input)
            loss_joint = loss_fn(joint_output, binary_label)

        if loss_asr is not None or loss_error is not None or loss_joint is not None:
            loss =  (loss_asr if loss_asr is not None else 0) + \
                self.alpha * (loss_error if loss_error is not None else 0) 
                # + 0.2 * (loss_joint if loss_joint is not None else 0)

        if not return_dict:
            return (loss, logits, error_logits) + outputs.hidden_states
            # return (loss, logits, error_logits) + outputs.hidden_states[_HIDDEN_STATES_START_POSITION:]

        return MultitaskWav2Vec2Output(
            loss=loss,
            logits=logits,
            error_logits=error_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


