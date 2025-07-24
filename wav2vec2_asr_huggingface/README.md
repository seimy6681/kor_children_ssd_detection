## Wav2Vec2CTC finetuning

### 실행방법
```
python asr_finetuning_xls-r.py
```

### ! 코드 내부에서 수정
- line 26 & 27 train 데이터와 test 데이터 경로 지정
- TrainingArguments 중 `output_dir` 수정
- line 168 seed range 수정


#### asr_finetuning_xls_extract_vocab.py
- Wav2Vec2CTC 학습에 필요한 vocabulary 생성파일입니다.

#### vocab_jrnl.json
- 위 파일 실행 결과입니다. finetuning시 vocab_jrnl.json을 로딩합니다.
