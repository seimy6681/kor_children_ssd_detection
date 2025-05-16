### source: https://joungheekim.github.io/2021/04/02/code-review/

import sys
import numpy as np
import torch
import os
import argparse
import soundfile as sf
import debugpy
import hangul_jamo
from hparams import defaults
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write
# import IPython.display as ipd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import librosa.display
# sys.path.append('/home/selinawisco/waveglow')
from denoiser import Denoiser
from glow import WaveGlow
import pandas as pd

# (1) 학습된 타코트론 모델 경로
tacotron2_checkpoint = '/data/selinawisco/kochild/tacotron2_models_clear_500/checkpoint_98000'  

# (2) 생성할 텍스트 데이터 프레임
csv_to_generate = "/data/selinawisco/kochild/filelists/sample.csv"
df = pd.read_csv(csv_to_generate)

# (3) 저장할 경로
output_path = "./generation"
os.makedirs(output_path, exist_ok=True)
os.makedirs(f'{output_path}/attention_maps')
os.makedirs(f'{output_path}/spectrograms')

'''
타코트론이 생성한 스팩트로그램을 저장하는 함수
'''
def save_spectrogram(mel_outputs_postnet, filename):
    
        """Saves the generated mel-spectrogram to an image file."""
        mel_spectrogram = mel_outputs_postnet[0].detach().cpu().numpy()  # Take the first element in the batch
        # np.save(f'/data/selinawisco/kochild/waveglow_data/mel_{filename}.npy', mel_spectrogram)
        
        # Plotting the mel-spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spectrogram, sr=22050, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        plt.tight_layout()

        # Save as image
        plt.savefig(filename)
        plt.close()
        # print(f"Spectrogram saved as {filename}")


'''
생성한 음성에 대한 timestep 별 attention map 을 저장하는 함수
'''
def plot_attention_map(alignments, filename):
    # Convert the alignment tensor to a NumPy array
    alignment_matrix = alignments[0].cpu().detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(alignment_matrix, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar()
    plt.xlabel("Decoder Time Step")
    plt.ylabel("Encoder Time Step (Input Text Position)")
    plt.title(filename)
    plt.tight_layout()
    plt.show()

    plt.savefig(filename)
    plt.close()
    # print(f"Attention map saved in {filename}")


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
    

def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    return model

'''
타코트론 checkpoint 로 스팩트로그램 생성후 Pretrained 된 WaveGlow checkpoint 로 Wav 파일 생성하는
Text To Speech (TTS) 클래스
'''
class Synthesizer:
    def __init__(self, tacotron_check):
        hparams = Struct(**defaults)
        hparams.n_mel_channels = 80
        hparams.sampling_rate = 22050
        self.hparams = hparams

        model = load_model(hparams)
        model.load_state_dict(torch.load(tacotron_check)['state_dict'])
        model.cuda().eval()
        self.tacotron = model

        # # loading pretrained waveglow
        self.waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
        
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/nvidia_waveglowpyt_fp32_20190306.pth",  # noqa: E501
            progress=False,
            map_location='cpu',
        )
        state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}

        self.waveglow.load_state_dict(state_dict)
        
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
        self.waveglow.cuda().eval()

        
        # 커스텀 finetuning 한 waveglow checkpoint 로딩 시:
        # ---------------------------------------------------------------------------------------------
        # sys.path.append("/home/selinawisco/waveglow") 
        # waveglow_path = "/data/selinawisco/waveglow/tacotron_500_checkpoints/waveglow_10000"  # Update with your actual checkpoint path
        # checkpoint = torch.load(waveglow_path, map_location="cuda")
        # print(checkpoint.keys())
        # n_mel_channels = 80
        # n_flows = 12
        # n_group = 8
        # n_early_every = 4
        # n_early_size = 2
        # WN_config = {
        #     'n_layers': 8,
        #     'n_channels': 512,
        #     'kernel_size': 3
        # }
        
        # waveglow = WaveGlow(n_mel_channels, n_flows, n_group, n_early_every, n_early_size, WN_config)  # Initialize the model
        # waveglow.load_state_dict(checkpoint['model'].state_dict()) # Load model weights
        # # waveglow.load_state_dict(torch.load(waveglow_path, map_location="cuda"))  # Load model weights
        
        # # waveglow.remove_weightnorm()  # Remove weight normalization
        # waveglow.cuda().eval()  # Move to GPU and set to evaluation mode
        # self.waveglow = waveglow
        # ---------------------------------------------------------------------------------------------

        self.denoiser = Denoiser(self.waveglow)
    

    '''
    주어진 텍스트로 음성 생성하는 함수
    '''
    def inference(self, text):

        assert isinstance(text, str), 'single text please'

        sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
        # print(sequence)
       
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

        mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron.inference(sequence)

        save_spectrogram(mel_outputs_postnet, f"{output_path}/spectrograms/tacotron2_spectrogram_{text}.png")

        # print(alignments.shape)
        # Call this function after inference
        plot_attention_map(alignments, f"{output_path}/attention_maps/tacotron2_Attention Map_{text}")
        plt.savefig("attention_map.png")
        with torch.no_grad():
            audio = self.waveglow.infer(mel_outputs_postnet, sigma=0.66)
        
        audio = audio[0].data.cpu().numpy()
        return audio, self.hparams.sampling_rate
    
    '''
    denoiser 가 추가된 TTS 함수
    '''
    def denoise_inference(self, text, sigma=0.666):

        assert type(text)==str, "텍스트 하나만 지원합니다."
        # 텍스트 토큰화
        sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
        # print(f'{sequence=}')
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        # 토큰화 된 텍스트를 타코트론 모델에 통과시킨다.
        mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron.inference(sequence)
        
        #스팩트로그램, 어텐션 맵 저장
        save_spectrogram(mel_outputs_postnet, f"{output_path}/spectrograms/tacotron_spectrogram_{text}.png")
        plot_attention_map(alignments, f"{output_path}/attention_maps/tacotron2_Attention Map_{text}")

        # WaveGlow 모델로 스팩트로그램을 WAV 음성파일로 변환
        with torch.no_grad():
            audio = self.waveglow.infer(mel_outputs_postnet, sigma)
            
        # WaveGlow Denoiser 통과
        audio_denoised = self.denoiser(audio, strength=0.01)[:, 0].cpu().numpy()
        return audio_denoised.reshape(-1), self.hparams.sampling_rate
    

synthesizer = Synthesizer(tacotron2_checkpoint)


#-----------------------------------------------------------------------------------------
# Example genertion (single word)
# sample_text = "머리"
# audio, sampling_rate = synthesizer.denoise_inference(sample_text) # TTS 함수 호출
# sf.write(f'{output_path}/{sample_text}_tts.wav', audio, sampling_rate)
# ----------------------------------------------------------------------------------------

'''
데이터 프레임에서 human_text 칼럼의 entry 를 tts 로 생성하는 함수
'''
def generate_TTS(df, out_dir):
    tts_csv_data = []

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # print(len(df))
    for idx, row in df.iterrows():

        text = row['human_text']
        target_text = row['target_text']
        path = f"{out_dir}/{text}({target_text})_{idx}.wav"
        print(idx)
        audio, sr = synthesizer.denoise_inference(text)
        sf.write(path, audio, sr)
        
        tts_csv_data.append(
            {
            'audio': path,
            'disease_type': '',
            'age':'',
            'gender':'',
            'subgroup':'',
            'id':'',
            'textgrid_text':'',
            'target_text': target_text,
            'human_text': text,
            'asr_text': '',
            'new_label': 1,
            'target_text_jamo': hangul_jamo.decompose(target_text),
            'human_text_jamo': hangul_jamo.decompose(text),
            }
        )
    
    tts_df = pd.DataFrame(tts_csv_data)
    tts_df.to_csv(f'{out_dir}/tts_{os.path.basename(csv_to_generate)}', index=False)

generate_TTS(df, output_path)
