import sys
import numpy as np
import torch
import os
import argparse
import soundfile as sf
import debugpy
import hangul_jamo
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt

sys.path.append('tacotron2/')  # Ensure Tacotron-2 module is accessible

from hparams import defaults
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write
from tqdm.notebook import tqdm

# Add HiFi-GAN path manually if it's not installed as a package
# sys.path.append('hifi-gan/')  # Update this to your actual HiFi-GAN path
# from models import Generator  # Import HiFi-GAN Generator model

def save_spectrogram(mel_outputs_postnet, filename):
    """Saves the generated mel-spectrogram to an image file."""
    mel_spectrogram = mel_outputs_postnet[0].detach().cpu().numpy()  # Take the first batch element

    # Plotting the mel-spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, sr=22050, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()

    # Save as image
    plt.savefig(filename)
    plt.close()
    print(f"Spectrogram saved as {filename}")

def plot_attention_map(alignments, title="Attention Map"):
    """Plots the Tacotron-2 attention map."""
    alignment_matrix = alignments[0].cpu().detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(alignment_matrix, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar()
    plt.xlabel("Decoder Time Step")
    plt.ylabel("Encoder Time Step (Input Text Position)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

class Struct:
    """Convert dictionary into an object with dot notation."""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_checkpoint(checkpoint_path, model):
    """Loads a checkpoint into a model."""
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    return model

class Synthesizer:
    def __init__(self, tacotron_check):
        # Load Tacotron-2
        hparams = Struct(**defaults)
        hparams.n_mel_channels = 80
        hparams.sampling_rate = 16000
        self.hparams = hparams

        model = load_model(hparams)
        model = load_checkpoint(tacotron_check, model)
        model.cuda().eval()
        self.tacotron = model

        # Load HiFi-GAN from NVIDIA TorchHub
        self.hifigan,_,self.denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan') # hifigan model, training setup dict, denoiser
        
        self.hifigan.cuda().eval()

    def inference(self, text):
        """Generate speech from text using Tacotron-2 and HiFi-GAN"""
        assert isinstance(text, str), 'Provide a single string as input.'

        # Convert text to sequence for Tacotron-2
        sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

        # Generate Mel Spectrogram from Tacotron-2
        _, mel_outputs_postnet, _, alignments = self.tacotron.inference(sequence)

        save_spectrogram(mel_outputs_postnet, "tacotron_spectrogram.png")
        print(f'{mel_outputs_postnet.shape=}')
        print(f'{alignments.shape=}')
        plot_attention_map(alignments, title="Tacotron2 Attention Map")
        plt.savefig("attention_map.png")

        # Convert Mel Spectrogram to Audio using HiFi-GAN
        with torch.no_grad():
            
            audio = self.hifigan(mel_outputs_postnet)
            print(f'{audio.shape=}')
            # audio = self.denoiser(audio).cpu().numpy()

            # audio = denoiser

        return audio, self.hparams.sampling_rate

# Paths to model checkpoints
tacotron2_checkpoint = "/data/selinawisco/kochild/tacotron2_models_clear_500/checkpoint_98000"

# Initialize Synthesizer with Tacotron-2 and HiFi-GAN
synthesizer = Synthesizer(tacotron2_checkpoint)

# Load test dataset
df = pd.read_csv('/home/selinawisco/sel-hub/asr/r07_test_3000_2.csv')

tts_csv_data = []

def generate_TTS(df, out_dir):
    """Generates TTS output for a dataset and saves results to CSV"""
    os.makedirs(out_dir, exist_ok=True)
    print(len(df))

    for idx, row in df.iterrows():
        text = row['human_text']
        target_text = row['target_text']
        path = f"{out_dir}/{text}({target_text})_{idx}.wav"
        print(f"Processing {idx}: {text}")

        audio, sr = synthesizer.inference(text)
        print(audio)
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        sf.write(path, audio.squeeze().astype(np.float32), sr)

        # Store metadata
        tts_csv_data.append({
            'audio': path,
            'disease_type': '',
            'age': '',
            'gender': '',
            'subgroup': '',
            'id': '',
            'textgrid_text': '',
            'target_text': target_text,
            'human_text': text,
            'asr_text': '',
            'new_label': 1,
            'target_text_jamo': hangul_jamo.decompose(target_text),
            'human_text_jamo': hangul_jamo.decompose(text),
        })

    # Save results to CSV
    tts_df = pd.DataFrame(tts_csv_data)
    tts_df.to_csv('/home/selinawisco/sel-hub/asr/r07.error_generated_3000_tacotron2_hifigan.csv', index=False)

generate_TTS(df, '/data/selinawisco/kochild/r07.error_generated_3000_tacotron2_hifigan')
