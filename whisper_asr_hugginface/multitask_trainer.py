from transformers import Seq2SeqTrainer
import torch

class MultitaskTrainer(Seq2SeqTrainer):
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None
    ):
        has_labels = all(k in inputs for k in ("labels", "binary_labels"))
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)

        # extract outputs
        loss = outputs.loss if has_labels else None
        logits = outputs.logits
        binary_logits = outputs.binary_logits

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        binary_labels = inputs["binary_labels"] if "binary_labels" in inputs else None

        return (loss, (logits, binary_logits), (labels, binary_labels)) # also return binary labels and logits for inference/validation
