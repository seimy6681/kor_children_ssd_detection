# Tacotron 2(NVIDIA + hccho2)
This is modified versrion of NVIDIA Tacotron2, considering korean preprocess method.
Please see [Blog Post](https://joungheekim.github.io/2021/04/01/code-review/) written in korean for details.

## Finetuning
- 실행 방법:
```
./run.sh
```
- 데이터셋
    타코트론 finetuning 을 위한 데이터셋은 디렉토리 안 `children_ssd_detection/augmentation/tacotron2/taco_train_data_clear.txt`(학습용) 과 `children_ssd_detection/augmentation/tacotron2/taco_test_data_clear.txt` (테스트용) 입니다. 학습 데이터는 r08.1_train.csv 단어 레벨 데이터셋에서 clear 레이블이 붙은 음성만 필터한 것입니다.
- 학습
    hparams.py 에서 학습할 때 이용할 모델 파라미터를 수정할 수 있습니다. 
    음성 파일 Generation 할 때의 모델 파라미터와 불러오는 체크포인트의 모델을 학습할때 이용한 파라미터가 동일해야 합니다.
    

## Reference
- Full Code : [NVIDIA Tacotron2](https://github.com/JoungheeKim/tacotron2)
- Korean Preprocess Code : [hccho2 Tacotron2](https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS)

## Checkpoint 위치
- Word level r08.1_train.csv 로 파인튜닝 된 모델은 NAS 서버 안
`
shared/kochild/kochild/tacotron2/checkpoint_98000
`
에 위치해 있습니다.

- TTS 음성생성 파일은 
`
tacotron2/0.gen_wav_file.py
`
입니다. 
학습된 타코트론 체크포인트와 pretrained WaveGlow 체크포인트를 불러와 텍스트를 wav 파일로 생성하는 TTS 코드입니다.

    (1) 위 Checkpoint 위치로 타코트론 체크포인트 경로를 설정해주세요.
    `
    tacotron2_checkpoint = '/data/selinawisco/kochild/tacotron2_models_clear_500/checkpoint_98000'  
    `
    (2) 생성할 텍스트 리스트가 포함 된 CSV 파일 경로를 설정해주세요 (생성되는 칼럼 이름은 human_text 로 설정되어 있습니다).

    - 실행 방법
        ```
        python 0.gen_wav_file.py
        ```

