# Pandas
열 중심 데이터 분석 API -> 입력 데이터를 처리/분석하는데 효과적

## 기본 개념

```python
import pandas as pd
```

1. `DataFrame` : 행 & 열이 포함된 관계형 데이터 테이블 (하나 이상의 `Series`와 각 `Series`의 이름이 포함)
2. `Series` : 하나의 열

### `Series` 만들기
```python
pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
```

`Series` 객체를 만든다.

### `DataFrame` 만들기

```python
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

pd.DataFrame({ 'City name': city_names, 'Population': population })
```

한 개 이상의 `Series` 객체를 생성하고 열 이름과 매핑되는 딕셔내리를 전달해서 만든다.

위 예제 코드에서는 `City name`이라는 이름의 열에 `['San Francisco', 'San Jose', 'Sacramento']` 데이터가, `Population`에 `[852469, 1015785, 485199]` 데이터가 있을 것이다.

즉, 출력결과는 다음과 같다:

```
City name	Population
0	San Francisco	852469
1	San Jose	1015785
2	Sacramento	485199
```

~~좋았어, 여기까지 이해했다!~~

### 데이터가 있는 파일 로드하기
데이터가 있는 파일 전체를 `DataFrame`으로 로드하는 경우가 많다고 한다.

```python
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
```

이렇게 읽어오는데, 이때 `pandas.read_csv`는 무엇일까? pandas-docs를 찾아봤다(무려 1344 페이지에 있었다).

CSV (comma-separated) 파일을 읽어서 데이터프레임 형태로 불러오는 함수인 것 같다. CSV 파일 형식에 대해서는 잘 모르지만 comma-separated니까 각각의 데이터가 콤마로 구분되는 모양과 상당히 유사한 구조를 가진 듯했다.

```python
import pandas as pd

df = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

print('[*] Using Dataframe.describe')
print(df.describe())

print('\n[*] Using Dataframe.head')
print(df.head())
```

`DataFrame.describe`는 데이터프레임의 데이터에 대한 통계를 이쁘게 표시해 주고, `DataFrame.head`는 데이터프레임에 있는 데이터 중 처음 몇 개 데이터만 보여준다.

```
[*] Using Dataframe.describe
          longitude      latitude  housing_median_age         ...            households  median_income  median_house_value
count  17000.000000  17000.000000        17000.000000         ...          17000.000000   17000.000000        17000.000000
mean    -119.562108     35.625225           28.589353         ...            501.221941       3.883578       207300.912353
std        2.005166      2.137340           12.586937         ...            384.520841       1.908157       115983.764387
min     -124.350000     32.540000            1.000000         ...              1.000000       0.499900        14999.000000
25%     -121.790000     33.930000           18.000000         ...            282.000000       2.566375       119400.000000
50%     -118.490000     34.250000           29.000000         ...            409.000000       3.544600       180400.000000
75%     -118.000000     37.720000           37.000000         ...            605.250000       4.767000       265000.000000
max     -114.310000     41.950000           52.000000         ...           6082.000000      15.000100       500001.000000

[8 rows x 9 columns]

[*] Using Dataframe.head
   longitude  latitude  housing_median_age         ...          households  median_income  median_house_value
0    -114.31     34.19                15.0         ...               472.0         1.4936             66900.0
1    -114.47     34.40                19.0         ...               463.0         1.8200             80100.0
2    -114.56     33.69                17.0         ...               117.0         1.6509             85700.0
3    -114.57     33.64                14.0         ...               226.0         3.1917             73400.0
4    -114.57     33.57                20.0         ...               262.0         1.9250             65500.0

[5 rows x 9 columns]
```

아래 코드처럼 `DataFrame.hist`를 사용하면 '한 열에서 값의 분포를 빠르게 검토할 수 있는' matplotlib 히스토그램이 그려진다(그래핑). 인수로 주어진 열 이름에 해당하는 값들의 분포가 나타나는 히스토그램이다. 

```python
import pandas as pd

df = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

df.hist('latitude')
```

~~이걸 이해하는데 오래 걸렸다.~~

## 데이터 액세스

아래처럼 다양한 방법으로 데이터프레임의 데이터에 액세스할 수 있다.

```python
import pandas as pd

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
cities['City name']
# 0    San Francisco
# 1         San Jose
# 2       Sacramento
cities['City name'][1]
# 'San Jose'
cities[0:2]
#        City name  Population
# 0  San Francisco      852469
# 1       San Jose     1015785
```

~~우와!~~

## lambda 표현식
원래는 '데이터 조작'에 대해서 설명이 나오는데 그 내용 중 `lambda`가 있었다.

필자는 람다를 불과 몇 달 전까지만 해도 '람바다'로 알고 있을 정도로 무식했고 람다만 보면 왠지 매우 복잡한 무언가를 아주 조그만 공간에 집약시켜 둔 듯한 끔찍한 느낌이 드는 `람다공포증`이 있었는데 이를 이 기회에 극복해보기로 했다. 그런데 별 거 아니여서 자괴감이 들고 괴로웠다.

```python
>>> (lambda x,y: x + y)(10, 20)
30
>>> (lambda x: x % 2)(1)
1
>>> (lambda x: x % 2)(2)
0
```

그냥 설명을 생략하겠다(grin).

## 데이터 조작

```python
import pandas as pd

population = pd.Series([852469, 1015785, 485199])
print(population / 1000)
# 0     852.469
# 1    1015.785
# 2     485.199
# dtype: float64
```

기본 산술 연산을 `Series`에 적용할 수도 있다.

위 코드는 `Series`의 모든 데이터를 1000으로 나눠서 출력한다.

```python
import numpy as np
import pandas as pd

population = pd.Series([852469, 1015785, 485199])
print(np.log(population))
# 0    13.655892
# 1    13.831172
# 2    13.092314
# dtype: float64
```

이렇게 넘파이 함수도 사용할 수 있다고 하는데 넘파이는 잘 모르니까 일단 패스하고 나중에 자세히 살펴봐야겠다.

```python
import pandas as pd

population = pd.Series([852469, 1015785, 485199])
print(population.apply(lambda val: val > 1000000))
# 0    False
# 1     True
# 2    False
# dtype: bool
```

`Series.apply`는 인수로 `lambda` 함수도 허용하는데, 위 코드에서는 인구가 `1000000`명을 넘는지에 대한 새 `Series`를 만들어서 출력한다.

```python
import pandas as pd

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
print(cities)
#        City name  Population  Area square miles  Population density
# 0  San Francisco      852469              46.87        18187.945381
# 1       San Jose     1015785             176.53         5754.177760
# 2     Sacramento      485199              97.92         4955.055147
```

이렇게 새로 `Series`도 추가할 수 있다고 한다.

~~와!!! 드디어 끝났다!~~ ~~응 실습이랑 색인 남았어~~

## 색인

### `DataFrame.index`
각 `Series` 항목 및 `DataFrame` 행에 해당하는 index 속성을 정의한다.

```python
import pandas as pd

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print(cities.index) # RangeIndex(start=0, stop=3, step=1)
print(city_names.index) # RangeIndex(start=0, stop=3, step=1)
```

### `DataFrame.reindex`
행의 순서를 재정렬할 때 사용한다.

```python
import pandas as pd

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print(cities.reindex([2, 0, 1]))
#        City name  Population
# 2     Sacramento      485199
# 0  San Francisco      852469
# 1       San Jose     1015785
```

아래처럼 랜덤으로 `DataFrame`을 섞는 것 역시 유용하게 사용된다고 한다.

```python
import pandas as pd

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print(cities.reindex(np.random.permutation(cities.index)))
```

그럼 이제 드디어 끝났으니 실습 문제를 풀어보자.

## 실습 #1
각 도시의 이름이 성인의 이름을 본뜬 경우 + 면적이 130제곱킬로미터보다 넓으면 `True`인 열을 추가하면 된다.

힌트 두 가지가 주어진다.

- 성인의 이름을 딴 도시의 이름은 `San`으로 시작한다.
- `bool`인 `Series`는 `logical and`를 실행할 때 `and` 대신 `&`를 사용한다.

`str.startswith()`로 문자열이 `San`인지 확인하고 `&`로 `Area square miles`의 값도 확인하는 방식으로 조건을 확인하면 될 것이다.

```python
cities['Is wide and has saint name'] = ((cities['Area square miles'] > 50) & cities['City name'].apply(lambda x: x.startswith('San')))
#        City name  Population  Area square miles  Is wide and has saint name
# 0  San Francisco      852469              46.87                       False
# 1       San Jose     1015785             176.53                        True
# 2     Sacramento      485199              97.92                       False
```

답안처럼 새로운 `Series` 이름도 `Is wide and has saint name`로 해줬다!

## 실습 #2
```
reindex 메서드는 원래 DataFrame의 색인 값에 없는 색인 값을 허용합니다. 메서드를 실행해보고 이런 값을 사용하면 어떤 결과가 나오는지 확인해보세요. 왜 이런 값이 허용된다고 생각하나요?
```

누락된 색인에 새 행을 추가하고 모든 열을 `NaN`으로 채운다.

```python
>>> cities.reindex([1, 2, 3, 4])
    City name  Population  Area square miles Is wide and has saint name
1    San Jose   1015785.0             176.53                       True
2  Sacramento    485199.0              97.92                      False
3         NaN         NaN                NaN                        NaN
4         NaN         NaN                NaN                        NaN
```
