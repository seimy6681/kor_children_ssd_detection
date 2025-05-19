import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2Encoder(nn.Module):
    def __init__(self,model_name="facebook/wav2vec2-base", output_dim=768):
        super(Wav2Vec2Encoder, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.output_dim = output_dim
        
    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state # (B, T, D)
        
        return last_hidden_state.mean(dim=1) # mean pooling over time dimension