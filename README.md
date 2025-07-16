## Korean Children Speech Sound Disorder Detection

### Wav2Vec2-CTC Forced alignment
- Wav2Vec2-CTC ASR 모델로 forced alignment 하는 코드 위치는 
```
whisper_asr_huggingface/util/forced_aligner.py
```
입니다. 코드 맨 아래 실행 코드을 실행하면 음소단위 데이터셋과 CSV가 저장됩니다. (주석 처리 해제)

### asr_ssd_main
- Wav2Vec2 ASR 모델 코드입니다. 자세한 정보는 asr_ssd_main/READ.md 참고해주세요.

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
