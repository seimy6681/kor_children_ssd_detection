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
    
    
class AgeEmbeddingWav2Vec2ForCTC(Wav2Vec2ForCTC):

    def __init__(self, config, main_arg=None):

        super().__init__(config)
        # self.age_classifier = nn.Sequential(
        #     nn.Linear(config.hidden_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, 9)
        # )
        
        self.age_embedding = nn.Embedding(num_embeddings=9, embedding_dim=config.hidden_size) # 10 unique age(2-10)
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
            **kwargs,
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

        if age_label is None:
            age_label = kwargs.get('age_label', None)
        if age_label is not None:
            ###embedding model   
            age_label = age_label.type(torch.LongTensor)-2         
            # age_label = age_label.unsqueeze(0).type(torch.LongTensor)-2 # [1]
            age_embedding = self.age_embedding(age_label.to(hidden_states.device)) # [B, 1, H]
            age_embedding_expanded = age_embedding.expand(hidden_states.shape[0], hidden_states.shape[1], self.config.hidden_size) # repeat age embedding across all time steps
            embedded_hidden_states = hidden_states + age_embedding_expanded

        logits = self.lm_head(embedded_hidden_states) # Hidden size H -> Vocab size V

        loss_asr = None
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

        if loss_asr is not None :
            loss = loss_asr if loss_asr is not None else 0
        if not return_dict:
            return (loss, logits) + outputs.hidden_states

        return MultitaskWav2Vec2Output(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )