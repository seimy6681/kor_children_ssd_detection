import pandas as pd
import os

# Load DataFrames
audio_df = pd.read_csv("/home/selinawisco/sel-hub/asr/r08.1_train.csv")  # Contains column "audio_path"
mel_df = pd.read_csv("/home/selinawisco/sel-hub/asr/waveglow_training_data.csv")      # Contains column "mel_path"

# Extract filenames without extensions
audio_df["filename"] = audio_df["audio"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
mel_df["filename"] = mel_df["npy"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

# Merge on filename to match spectrogram and audio files
paired_df = pd.merge(audio_df, mel_df, on="filename", how="inner")

paired_df.to_csv('/home/selinawisco/sel-hub/asr/paired_waveglow_training_data.csv')