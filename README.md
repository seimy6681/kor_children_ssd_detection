## Korean Children Speech Sound Disorder Detection

### asr_ssd_main
- Wav2Vec2 ASR 모델 코드입니다. 자세한 정보는 asr_ssd_main/READ.md 참고해주세요.

### augmentation
- tacotron2 : tacotron2+waveglow TTS 시스템
- mixing_and_others/age_mixing.ipynb : 인터스피치 24 나이 믹싱

### whisper_asr_huggingface
- Whisper ASR finetuning 스크립트 `0.whisper_asr_finetuning.py`
- Whisper ASR inference (CER & UAR 계산) 스크립트 `0.whisper_asr_transcribe.py`

### datasets
- csv 데이터셋
