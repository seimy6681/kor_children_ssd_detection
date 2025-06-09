import pandas as pd

def extract_vocab(train_df, test_df, transcript_col="label"):
    all_phonemes = set()

    for df in [train_df, test_df]:
        for jamo_seq in df[transcript_col]:
            all_phonemes.update(jamo_seq)  # assumes jamo_seq is a string of phoneme chars
    
    return sorted(all_phonemes)
