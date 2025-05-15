# asr_ssd_main
Korean Children Speech Sound Disorder Detection using ASR

This project aims to develop an effective Automatic Speech Recognition system that generates an accurate transcription of a child's speech to be used to diagnose whether he or she has a Speech Sound Disorder (SSD).

## Base ASR Model
We use the baseline model Wav2Vec2CTC for ASR and aim to improve its Character Error Rate (CER) on our Korean SSD Dataset to leverage its ability to capture mispronunciations in Korean.

## Custom Models
To improve the model's ability to recognize correct pronounciation against its incorrect pronounciations, we apply multitask learning by attaching an auxilary classification head to the model.
- binary multitask
- dual_multitask
- multitask



