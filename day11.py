import numpy as np

v = np.array([1, 3, -9, 2])
# v = np.array((1, 3, -8, 4))
print(v, v.ndim,v.shape,v.dtype,v.strides)

b = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])
print(b,b.ndim,b.shape,b.dtype, b.strides)

c = np.array([[[1, 2, 3, 4],[5, 6, 7, 8]],[[1, 2, 3, 4],[9, 10, 11, 12]]])
print(c, c.ndim,c.shape,c.dtype,c.strides)

# ndim : 배열의 차원 수를 나타내는 속성
# shape : 배열의 차원과 크기를 나타내는 튜플(tuple)형태의 속성
# dtype : 타입 확인
# stride : 간격에 대한 정보