### Contrastive Learning for Korean Children SSD detection

## 모델 구조
 - TripletLoss 손실함수를 이용하여 anchor 음성을 를 랜덤으로 선택이 되면, positive pair 는 anchor 의 발음 맞춤/틀림 레이블과 같고 anchor의 타겟단어가 같은 다른 음성으로, negative pair는 타겟단어는 일치하지만 반대의 레이블인 음성으로 생성되어, positive pair 의 임베딩 거리를 좁히고, negative pair 의 임베딩 거리를 떨어놓게 하고자 하는 모델입니다.
 - Wav2Vec2 Encoder 를 통과한 hidden states 피쳐를 이용하고 파이토치로 구현 한 코드 입니다.

### 학습
- 실행 방법:
    ```
    run.sh
    ```
    -- model_path: 저장할 모델 이름

- 모델 optimizer 는 Adam, 스케줄러는 StepLR을 사용합니다.

### Validation 및 Inference
- trainset 에서 타겟 단어 별로 정상음성만 모아서 평균 reference_embedding 을 만든 후, validation set 의 각각 음성을 돌면서 0/1 레이블 프리딕션 할 때 최적인 threshold(이 이상이면 정상 임베딩과 멀어 비정상으로 판단될 수치)을 계산하고, 이 threshold 와 테스트 타임 때 각 오디오와 referene