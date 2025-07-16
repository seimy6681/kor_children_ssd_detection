## Korean Children Speech Sound Disorder Detection

### Wav2Vec2-CTC Forced alignment
- Wav2Vec2-CTC ASR 모델로 forced alignment 실행하는 코드 위치는 
```
whisper_asr_huggingface/util/generate_forced_alignment_data.py
```
입니다. 파이토치 튜토리얼 기반인 `whisper_asr_huggingface/util/forced_aligner.py` 를 불러와
실행하며, 코드 내부에서 지정한 각각의 경로로 음소단위 데이터셋과 CSV를 저장하는 구조입니다. 

### asr_ssd_main
- Wav2Vec2 기반 ASR 모델 코드입니다. Hugging Face의 fine-tuning 스크립트를 확장하여, 다양한 모델과 손실 함수를 선택할 수 있도록 main에서 argument를 받아 처리합니다. 선택된 구성에 따라 train_asr.py의 train() 함수가 main에서 호출되어 학습이 됩니다.
학습은 ./run.sh 스크립트를 실행하면 시작되며, 학습된 모델은 inference.py 안에서 저장된 모델 경로를 MODEL_PATH를 지정하고 실행하면 CER 및 UAR metrics를 확인할 수 있습니다. 자세한 내용은 asr_ssd_main/README.md를 참고해주세요.

### augmentation
- tacotron2 : tacotron2+waveglow TTS 시스템
- mixing_and_others/
    - age_mixing.ipynb : 인터스피치 24 나이 믹싱
    - phoneme_mixing.ipynb: 음소 단위 믹싱 (강제정렬 이용)

### whisper_asr_huggingface
- Whisper ASR finetuning 스크립트 `0.whisper_asr_finetuning.py`
- Whisper ASR inference (CER & UAR 계산) 스크립트 `0.whisper_asr_transcribe.py`

### datasets
- csv 데이터셋
