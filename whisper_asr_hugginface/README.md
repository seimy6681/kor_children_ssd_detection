- Wav2Vec2-CTC based forced alignment 실행 코드 위치는
```
whisper_asr_huggingface/util/generate_forced_alignment_data.py
```
입니다. 
파이토치 튜토리얼 기반인 `whisper_asr_huggingface/util/forced_aligner.py` 를 불러와
실행하며, 코드 내부에서 지정한 각각의 경로로 음소단위 데이터셋과 CSV를 저장하는 구조입니다. 


### Whisper + MTL 실행방법
```
python 0.multitask_whisper_asr_finetuning.py
```
- whisper_models/multitask_whisper_for_condtional_generation.py 를 이용하고, binary_logits을 함께 반환하는 ./multitask_trainer.py를 이용합니다.

### Whisper + Phoneme MTL 실행방법
```
python 0.phoneme_mtl_whisper_asr_finetuning.py
```
- whisper_models/phoneme_mtl_whisper_for_condtional_generation.py 를 이용
- ! forced alignment 를 통해 각 샘플의 음소별 (start,end) time step 정보가 필요합니다. 
    * 기존 단어단위 데이터셋으로 `util/generate_forced_aligned_intervals.py`코드 실행하여 단어 안 음소별로 whisper 프레임을 저장하는 칼럼 'start_sample_whisper'와 'end_sample_whisper' 를 추가 후 finetuning 실행
-  forward() 안에서 encoder_hidden을 나누고 mean pooling 후 phoneme classifier 통과