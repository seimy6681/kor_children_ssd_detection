# This class extends the WhisperForConditionalGeneration with two things:
#  (1) encoder output is obtained from the custom FusedWhisperEncoder which extends WhisperEncoder with a fused attention mechanism
#  (2) the fused encoder output is used for multitask learning, ASR and Binary Classification

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

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from whisper_models.fused_whisper_encoder import FusedWhisperEncoder

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
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration

from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
from transformers import WhisperModel, WhisperPreTrainedModel
from transformers.models.whisper.modeling_whisper import shift_tokens_right
from transformers import GenerationConfig
from dataclasses import dataclass

@dataclass
class MultitaskSeq2SeqLMOutput(Seq2SeqLMOutput):
    """
    Custom output for multitask Seq2Seq Whisper with both ASR logits and binary classification logits.
    """
    binary_logits: Optional[torch.FloatTensor] = None  # Binary classification logits



# @add_start_docstrings(
#     "The Whisper Model with a language modeling head. Can be used for automatic speech recognition.",
#     WHISPER_START_DOCSTRING,
# )
class FusedMultitaskWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        # Initialize the binary classifier for multitask learning
        self.binary_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1)
        )

        # fusion layers moved to whisper encoder to avoid redundant attention calculation per token in generate()
        # self.second_attention_layer = nn.Sequential(
        #     nn.Linear(config.d_model, config.d_model),
        #     nn.Tanh(),
        #     nn.Linear(config.d_model, 1),
        #     nn.Softmax(dim=1)
        # )
        # self.fusion_layer = nn.Linear(config.d_model * 2 , config.d_model)

        self.model.encoder = FusedWhisperEncoder(config) # replace the WhisperEncoder with custom FusedWhisperEncoder

        self.generation_config = GenerationConfig

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
        binary_labels: Optional[torch.LongTensor] = None,
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
        
        # print('1')

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        # print('2')

        # (1) get last encoder hidden state to pass through fused attention
        if encoder_outputs is None: # if model is training, encoder_outputs must be obtained through the encoder
            encoder_outputs = self.model.encoder(input_features, return_dict=True)
        # if model is evaluating, model.generate() precomputes encoder_outputs, so skip calling the encoder (input_features will be None)
        fused_encoder_hidden = encoder_outputs.last_hidden_state # (B, T, D)

        # # (2) fused attention
        # attention_scores = self.second_attention_layer(encoder_hidden) # (B, T, 1)
        # attended_states = attention_scores * encoder_hidden
        # concat_states = torch.cat([encoder_hidden, attended_states], dim=-1) # concatenate along the last dimension D (B, T,)
        # fused_states = self.fusion_layer(concat_states)
        
        # print('3')


        # fused states decoder 통과
        decoder_outputs = self.model.decoder(
          input_ids=decoder_input_ids,
          encoder_hidden_states=fused_encoder_hidden,
        #   encoder_hidden_states=fused_states,
        #   decoder_attention_mask=decoder_attention_mask,
          use_cache=use_cache,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=True,
        )
        
        lm_logits = self.proj_out(decoder_outputs.last_hidden_state) # (B, decoder_seq_len (i.e. decoder_input_ids.size(1)), V(whisper_small=51865))

        binary_logits=None

        # print('4')
        if attention_mask is not None:
            # (1) grab attention_mask to zero out padding tokens timesteps
            mask = attention_mask[:, ::2] # grab every 2nd input frame to match whisper downsampling in the whisper's encoder conv1 stride of 2
            mask = mask.unsqueeze(-1).type_as(fused_encoder_hidden) # (B, T, 1)
            # (2) apply attention mask to the fused_hidden_states
            masked_hidden = fused_encoder_hidden * mask # (B, T, D)
            # (3) sum hidden across all time steps
            sum_hidden = masked_hidden.sum(dim=1) # (B, D)
            # (4) get the number of valid tokens per sample (length)
            lengths = mask.sum(dim=1).clamp(min=1) # length of sample
            # (5) divide each sample's summed hidden states by the length
            encoder_output = sum_hidden / lengths # (B,D)
            # (6) pass it through the classifier
            binary_logits = self.binary_classifier(encoder_output).squeeze(-1)
        # print('5')
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
        
        binary_loss = None
        total_loss = None
        if binary_labels is not None:
            # get the binary label
            loss_fct_binary = BCEWithLogitsLoss()
            binary_labels = binary_labels.to(lm_logits.device)
            binary_loss = loss_fct_binary(binary_logits, binary_labels)

        if labels is not None and binary_labels is not None:
            total_loss = loss + 0.15 * binary_loss

        if not return_dict:
            return (total_loss, lm_logits, binary_logits)
            # output = (lm_logits, binary_logits) + outputs[1:]
            # return ((total_loss,) + output) if total_loss is not None else output
        # print('6')
        return MultitaskSeq2SeqLMOutput(
            loss=total_loss,
            logits=lm_logits,
            # past_key_values=outputs.past_key_values,
            # decoder_hidden_states=outputs.decoder_hidden_states,
            # decoder_attentions=outputs.decoder_attentions,
            # cross_attentions=outputs.cross_attentions,
            # encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            # encoder_hidden_states=outputs.encoder_hidden_states,
            # encoder_attentions=outputs.encoder_attentions,
            binary_logits=binary_logits # include the binary logits to output for compute_metrics()
        )