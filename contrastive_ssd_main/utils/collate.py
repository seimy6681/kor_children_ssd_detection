from transformers import Wav2Vec2Processor
import numpy as np
import torch


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def collate_fn(batch):
    """
    Custom collate function for batching triplets:
    (anchor, positive, negative) → processed inputs
    """
    
    anchors, positives, negatives = zip(*batch) # unpacks by zipping the first group anchors, positives and then negatives in their each zipped gorup
    
    # flatten all audio into a single list
    all_audio = list(anchors + positives + negatives)
    # for i, audio in enumerate(all_audio):
    #     if torch.isnan(audio).any():
    #         print(f" NaN in sample {i}")
    # all_audio = [x.tolist() for x in all_audio]
    all_audio = [np.asarray(x).astype(np.float32) for x in all_audio]
    
            
    # for i, x in enumerate(all_audio):
    #     if not isinstance(x, torch.Tensor):
    #         print(f" Sample {i} is not a tensor: {type(x)}")
    #     elif x.ndim != 1:
    #         print(f" Sample {i} is not 1D: shape = {x.shape}")

    # assert all([isinstance(x, torch.Tensor) for x in all_audio]), "💥 Non-tensor input detected!"
    # assert all([x.ndim == 1 for x in all_audio]), "💥 Found non-1D audio!"

    # for i, x in enumerate(all_audio):
    #     print(f"Sample {i}: type = {type(x)}, shape = {x.shape if isinstance(x, torch.Tensor) else 'N/A'}")

    processed = processor(
        all_audio, # list of torch.Tensor waveforms
        sampling_rate=16000,
        return_tensors="pt", # return pytorch tensors
        return_attention_mask=True,
        padding=True # pad all audio in the batch to the longest sample
    )
    
    B = len(anchors)
    
    return {
        "input_values": processed.input_values[:B],
        "pos_values": processed.input_values[B:2*B],
        "neg_values": processed.input_values[2*B:],
        "attention_mask": processed.attention_mask[:B],
        "pos_mask": processed.attention_mask[B:2*B],
        "neg_mask": processed.attention_mask[2*B:],
    }
    
    
    