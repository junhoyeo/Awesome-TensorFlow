# 텐서플로우 첫걸음(First steps with TensorFlow)

## 설정
```python
from __future__ import print_function
import math

# from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
```

필요한 라이브러리를 가져온다.

```python
california_housing_dataframe = pd.read_csv("https://dl.google.com/mlcc/mledu-datasets/california_housing_train.csv", sep=",")
```

`pandas.read_csv`를 사용해서 데이터세트를 로드해서 데이터프레임으로 가져온다.

```python
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
print(california_housing_dataframe)
```

- 확률적 경사하강법 -> 데이터를 무작위로 추출
- 일반적 학습률 범위에서 학습 -> `median_house_value`를 천 단위로 조정

## 데이터 조사
데이터를 다루기 전 살펴보는 습관을 가지자 -> `DataFrame.describe`로 몇 가지 유용한 통계를 살펴봄

```python
print(california_housing_dataframe.describe())
```

## 첫 번째 모델 만들기

- 입력 특성으로 `total_rooms`(해당 지역의 전체 방 수) 사용 -> 라벨 `median_house_value`에 대한 예측을 시도
- 모델 학습에는 Estimator API가 제공하는 LinearRegressor 인터페이스를 사용

### 1단계: 특성 정의 및 특성 열 구성
학습 데이터 -> 텐서플로우 : 각 특성의 데이터 유형을 지정해야 함

- 범주형 데이터 : 텍스트
- 수치 데이터 : 정수 또는 실수 -> 숫자로 취급

텐서플로우에서 특성의 데이터 유형을 지정 -> 특성 열이라는 구조체를 사용 : 특성 데이터에 대한 설명만 저장되고 특성 데이터 자체는 포함하지 X

```python
# 입력 특성 정의
my_feature = california_housing_dataframe[["total_rooms"]]
print(my_feature)
#        total_rooms
# 10380       2074.0
# 11827       1564.0
# 9133        3415.0
# 4467        2302.0
# 4857        2168.0
# ...            ...
# 5432        1576.0
# 16103       2106.0
# 362         1978.0
# 5365         647.0
# 4202        2127.0

# [17000 rows x 1 columns]
```

`california_housing_dataframe`에서 `total_rooms` 데이터를 추출해서 `my_feature`에 저장한다.

```python
# 입력 특성 열 구성
feature_columns = [tf.feature_column.numeric_column("total_rooms")]
print(feature_columns)
# [_NumericColumn(key='total_rooms', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)]
```

`numeric_column`으로 수치 데이터 유형의 특성 탭을 새로 정의한다.

### 2단계: 타겟 정의

타겟 라벨인 `median_house_value`를 `california_housing_dataframe`에서 가져와 정의한다.

```python
targets = california_housing_dataframe["median_house_value"]
print(targets)
# 10380   400.0
# 11827   150.0
# 9133     84.4
# 4467    273.8
# 4857    300.9
#          ... 
# 5432    261.8
# 16103   500.0
# 362     170.1
# 5365    162.5
# 4202    188.5
# Name: median_house_value, Length: 17000, dtype: float64
```

### 3단계 : LinearRegressor 구성

LinearRegressor로 선형 회귀 모델 구성 -> SGD 구현하는 `GradientDescentOptimizer` 사용해 학습

- `learning_rate` : 경사 단계의 크기 조절
- `clip_gradients_by_norm` : 경사 제한 적용

```python
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)
```

### 4단계: 입력 함수 정의

### 5단계 : 모델 학습
```python
_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=100
)
```

### 6단계: 모델 평가
