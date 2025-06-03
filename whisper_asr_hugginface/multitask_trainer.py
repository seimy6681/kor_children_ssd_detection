 # Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers import Seq2SeqTrainer
import torch
import torch.nn as nn
from typing import Union, Any, Optional

class MultitaskTrainer(Seq2SeqTrainer):

    def __init__(self, config=None, data_args=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.data_args = kwargs.get("data_args", None)
        self.config = self.model.config

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]]=None,
    )-> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
        "max_length": self.data_args.val_max_target_length
        if self.data_args is not None
        else self.config.max_length,
        "num_beams": self.data_args.eval_beams if self.data_args is not None else self.config.num_beams,
        }

        # has_labels = all(k in inputs for k in ("labels", "binary_labels"))
        binary_labels = inputs.pop("binary_labels", None)

        if self.args.predict_with_generate and not self.args.prediction_loss_only:
            generated_tokens = self.model.generate(
                inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])


        labels = inputs.pop("labels")
        with torch.no_grad():
            outputs = model(**inputs, labels=labels, binary_labels=binary_labels)


        loss = outputs.loss.mean().detach()
        if self.args.prediction_loss_only:
            return (loss, None, None, None)

        logits = generated_tokens if self.args.predict_with_generate else outputs.logits
        if prediction_loss_only:
            return (loss, None, None)

        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        
            binary_logits = outputs.binary_logits.detach().cpu() if hasattr(outputs, "binary_logits") else None

        return (loss, (logits, binary_logits), (labels, binary_labels)) # also return binary labels and logits for inference/validation