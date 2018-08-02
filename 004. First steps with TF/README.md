# 텐서플로우 첫걸음 : 도구(Toolkit)

## 에스티메이터(tf.estimator)

- 고수준 OOP API
- 코드 행 수가 줄어듦
- scikit-learn API와 호환

# 텐서플로우 첫걸음: 프로그래밍 실습(Programming Exercises)

## 실습

1. Pandas : 데이터 분석, 모델링
2. 텐서플로우 첫걸음 : 선형 회귀
3. 합성 특성과 이상점

그냥 문제만 풀어보려고 했는데 도저히 사람이 할 게 안되서 따로 하위 디렉토리 만들어서 공부합니다.

## 초매개변수

- `steps` : 총 학습 반복 횟수, 한 단계/한 배치의 손실 계산 후 이 값을 이용하여 모델의 가중치를 한 번 수정
- `batch_size` : 하나의 단계와 관련된 배치(예시)의 수

```python
(학습하는 예시의 수) = batch_size * steps
```

## 변수
`periods` : 보고의 세부사항 제어

```python
(한 단계에서 학습하는 예시의 수) = (batch_size * steps) / periods
```
