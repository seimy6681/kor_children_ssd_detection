## Korean Children Speech Sound Disorder Detection

Different techniques using Audio Classification and Automatic Speech Recognition for the automatic detection of mispronunciation disorders in Speech Sound Disorders.
Custom code adaptation of Transformers & Pytorch implementations
Here we show the different methodlogies by directory.

### Wav2Vec2-CTC Forced alignment
- CTC-based forced alignment is located in `whisper_asr_huggingface/utilwhisper_asr_huggingface/util`

### asr_ssd_main
- Wav2Vec2-CTC Finetuning Scripts
- Wav2Vec2-CTC Custom models. can be found in `asr_ssd_main/models`
    - ASR + Age classifier multitask learning model
    - ASR + Age embedding model
    - ASR + Custom Loss weighting by Age
    - ASR + Fused Attention for weak supervision + Binary Classification Multitask Model (presented on our Interspeech 2025 paper "Multitask 
    Learning with Fused Attention for Improved ASR and Mispronunciation Detection in Children's Speech Sound Disorders)
    - ASR + Adversarial Learning with Age
    . . .
- Wav2Vec2 기반 ASR 모델 코드입니다. Hugging Face의 fine-tuning 스크립트를 확장하여, 다양한 모델과 손실 함수를 선택할 수 있도록 main에서 argument를 받아 처리합니다. 선택된 구성에 따라 train_asr.py의 train() 함수가 main에서 호출되어 학습이 됩니다.
학습은 ./run.sh 스크립트를 실행하면 시작되며, 학습된 모델은 inference.py 안에서 저장된 모델 경로를 MODEL_PATH를 지정하고 실행하면 CER 및 UAR metrics를 확인할 수 있습니다. 자세한 내용은 asr_ssd_main/README.md를 참고해주세요.

### augmentation
- tacotron2 : tacotron2+waveglow Text-to-Speech (TTS) System
- mixing_and_others/
    - age_mixing.ipynb : Age-based audio mixing (presented in our Interspeech 2024 paper here: https://www.isca-archive.org/interspeech_2024/kim24q_interspeech.html) 인터스피치 24 나이 믹싱
    - phoneme_mixing.ipynb: Phoneme level mixing for Data augmentation using CTC basec forced alignment. 음소 단위 믹싱 (강제정렬 이용)

### contrastive_ssd_main
- pytorch based implementation of contrastive learniing framework for audio classification
### whisper_asr_huggingface
- Whisper ASR & Audio classification model finetuning scripts & various custom models
- Wav2Vec2-CTC Custom models. can be found in `whisper_asr_huggingface/whisper_models`
- Available Custom models include:
    - ASR + Binary Classification Multitask Model
    - Fused attention + Multitask Model (presented on our Interspeech 2025 paper "Multitask Learning with Fused Attention for Improved ASR and Mispronunciation Detection in Children's Speech Sound Disorders)
    - ASR + Phoneme classification Multitask model (using forced alignment)
