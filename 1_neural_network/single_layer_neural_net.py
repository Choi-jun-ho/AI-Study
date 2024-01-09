import numpy as np
import matplotlib.pyplot as plt
import activaton

X = np.arange(-1.0, 1.0, 0.2) # 원소수는 10개
Y = np.arange(-1.0, 1.0, 0.2)

Z = np.zeros((10, 10)) # 출력값을 저장할 10X10 그리드

# 가중치: 뉴런의 흥분 조건 설정, 그리드 회전에 대한 영향력
w_x = 2.5
w_y = 3.0

bias = 0.1 # 편향: 뉴런을 흥분 시키는 정도, sigmoid를 사용시 그리드가 shift 됨

# 그리드맵의 각 그리드별 뉴런의 연산
for i in range(10):
    for j in range(10):

        # 입력과 가중치 곱의 합 + 편향
        u = X[i]*w_x + Y[j]*w_y + bias

        # 그리드맵에 출력값저장
        y = activaton.sigmoid_function(u)
        Z[j][i] = y

plt.imshow(Z, "gray", vmin = 0.0, vmax = 1.0)
plt.colorbar()
plt.show()
        


