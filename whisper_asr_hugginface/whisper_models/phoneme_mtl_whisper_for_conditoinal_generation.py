# We extend the WhisperForConditionalGeneration to add a wav2vec2 based forced alignment and phoneme classifier.

# coding=utf-8
# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Whisper model."""
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import wandb
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
from transformers import WhisperModel, WhisperPreTrainedModel
from transformers.models.whisper.modeling_whisper import shift_tokens_right
from transformers import GenerationConfig
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
from dataclasses import dataclass

@dataclass
class CustomSeq2SeqLMOutput(Seq2SeqLMOutput):
    """
    Custom output 
    """
    phoneme_loss: Optional[torch.FloatTensor] = None  # Binary classification logits


class PhonemeMtlWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    generation_config_class = GenerationConfig
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        # self.phoneme_classifier = nn.ModuleDict({
        #         str(i): nn.Linear(config.d_model, config.num_jamos) for i in range(config.num_jamos) # number of total unique characters
        #     })
        self.phoneme_classifier = nn.Linear(config.d_model, config.num_jamos)
        self.alpha = config.alpha
        
        
    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.model.encoder._freeze_parameters()

    # @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        phoneme_intervals: Optional[torch.LongTensor] = None,
        phoneme_targets: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False  # avoid memory growth

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.proj_out(outputs[0])
        encoder_hidden = outputs.encoder_last_hidden_state # using the last encoder layer output # (B, T, D), mean pooling across time steps (B, T, D)
        
        loss = None
        phoneme_loss = 0
        total_phoneme_count = 0
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.transpose(1,2), labels)
            # loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if phoneme_intervals is not None and phoneme_targets is not None:
            for b in range(encoder_hidden.shape[0]): # B
                for i in range(phoneme_targets.shape[1]): # L (length of jamos/phonemes)
                    target_phoneme = phoneme_targets[b,i].to(lm_logits.device) # current jamo in current sample b
                    if target_phoneme == -100: # reached PAD token
                        continue
                    start_frame = phoneme_intervals[b, 2*i] # current jamo's start frame
                    end_frame = phoneme_intervals[b, 2*i+1]
                    
                    if end_frame <= start_frame or end_frame > encoder_hidden.shape[1]:
                        continue
                    frames_phoneme = encoder_hidden[b,start_frame:end_frame] # (F, D) slicing encoder output using the start and frame for current jamo
                    pooled = frames_phoneme.mean(dim=0)
                    # phoneme_logits = self.phoneme_classifier[str(target_phoneme.item())](pooled.unsqueeze(0))  
                    phoneme_logits = self.phoneme_classifier(pooled.unsqueeze(0))  # (1, num_jamos)
                    curr_phoneme_loss = F.cross_entropy(phoneme_logits, target_phoneme.unsqueeze(0))
                    phoneme_loss += curr_phoneme_loss
                    total_phoneme_count += 1

        if loss is not None and phoneme_loss is not None:
            if total_phoneme_count > 0:
                phoneme_loss = (phoneme_loss / total_phoneme_count)
                loss = loss + self.alpha * phoneme_loss
                wandb.log({
                    "phoneme_loss": phoneme_loss
                })

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CustomSeq2SeqLMOutput(
            loss=loss, 
            phoneme_loss = phoneme_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            # decoder_hidden_states=outputs.decoder_hidden_states,
            # decoder_attentions=outputs.decoder_attentions,
            # cross_attentions=outputs.cross_attentions,
            # encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            # encoder_hidden_states=outputs.encoder_hidden_states,
            # encoder_attentions=outputs.encoder_attentions,
        )