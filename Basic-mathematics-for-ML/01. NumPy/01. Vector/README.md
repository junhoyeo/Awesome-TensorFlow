# 벡터(Vector)
~~배그에서 피해량이 가장 많은 SMG~~

- 벡터 공간에서의 **원소**를 표현
- 각 원소의 데이터 타입이 동일
- `np.array`

## 행백터(row vector)
```python
import numpy as np
x = np.array([[1, 2, 3]])
print(x) # [[1 2 3]]
print(x.shape) # (1, 3)
# 1 * 3
```

## 열백터(column vector)
```python
import numpy as np
x = np.array([[1], [2], [3]])
print(x) 
# [[1]
#  [2]
#  [3]]
print(x.shape) # (3, 1)
# 3 * 1
```
