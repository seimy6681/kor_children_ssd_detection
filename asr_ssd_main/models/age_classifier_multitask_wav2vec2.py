import torch
from torch import nn
import numpy as np
from transformers import Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional, Union, Tuple
from dataclasses import dataclass
@dataclass
class MultitaskWav2Vec2Output(CausalLMOutput):
    # age_loss: Optional[torch.FloatTensor] = None
    age_logits: Optional[torch.LongTensor] = None
    
    
class AgeClassifierWav2Vec2ForCTC(Wav2Vec2ForCTC):

    def __init__(self, config, main_arg=None):

        super().__init__(config)
        # self.age_classifier = nn.Sequential(
        #     nn.Linear(config.hidden_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, 9)
        # )

        self.age_classifier = nn.Linear(config.hidden_size, 9) # 10 unique age(2-10)
        self.alpha = main_arg.multitask_alpha if main_arg is not None else 0.3

    def forward(
            self,
            input_values: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
            age_label: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=return_dict
        )

        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states) # Hidden size H -> Vocab size V

        pooled_hidden_states = hidden_states.mean(dim=1)
        print(pooled_hidden_states.shape)
        age_logits = self.age_classifier(pooled_hidden_states)

        loss_asr = None
        loss_age = None
        loss = None

        if labels is not None:
            attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1).to(torch.long))

            labels_mask = labels >= 0    # tensor([true, true, false], [false, true, true])
            target_lengths = labels_mask.sum(-1)        # true is 1, false is 0
            flattened_targets = labels.masked_select(labels_mask)

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

        if age_label is not None:

            loss_fn = nn.CrossEntropyLoss()
            age_label = age_label.squeeze(1).type(torch.LongTensor)
            loss_age = loss_fn(age_logits, age_label.to(age_logits.device))

        if loss_asr is not None or loss_age is not None:
            loss = (loss_asr if loss_asr is not None else 0) + self.alpha * (loss_age if loss_age is not None else 0)
        
        if not return_dict:
            return (loss, logits, age_logits) + outputs.hidden_states

        return MultitaskWav2Vec2Output(
            loss=loss,
            logits=logits,
            # age_loss = loss_age,
            age_logits=age_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )