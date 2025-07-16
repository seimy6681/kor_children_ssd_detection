## Korean Children Speech Sound Disorder Detection

### Wav2Vec2-CTC Forced alignment
- Wav2Vec2-CTC ASR 모델로 forced alignment 실행하는 코드 위치는 
```
whisper_asr_huggingface/util/generate_forced_alignment_data.py
```
입니다. 파이토치 튜토리얼 코드 기반인 `whisper_asr_huggingface/util/forced_aligner.py` 를 불러와 불러
실행한 후 음소단위 데이터셋과 CSV를 저장헙니다. 

### asr_ssd_main
- Wav2Vec2 ASR 모델 코드입니다. 허깅페이스 finetuning script 에 여러 모델과 손실 함수 등을 추가하면서 main 에서 argument를 받고, argument 에 따라 해당 모델과 손실 함수등을 로딩하여 train_asr.py 안 train() 을 부르는 구조입니다. 학습 후 inference 는 저장된 모델 경로로 `inference.py` 에서 따로 실행합니다.
학습은 `./run.sh` 실행하면 시작됩니다. 자세한 정보는 asr_ssd_main/READ.md 참고해주세요.

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
