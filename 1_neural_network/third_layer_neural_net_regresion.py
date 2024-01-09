import numpy as np
import matplotlib.pyplot as plt
import activaton

X = np.arange(-1.0, 1.0, 0.2) # 원소수는 10개
Y = np.arange(-1.0, 1.0, 0.2)

Z = np.zeros((10, 10)) # 출력값을 저장할 10X10 그리드

# 가중치: 뉴런의 흥분 조건 설정, 그리드 회전에 대한 영향력
w_im = np.array([[4.0, 4.0],
                 [4.0, 4.0]]) # 은닉층 2x2 행렬
w_mo = np.array([[1.0],
                 [-1.0]]) # 출력층 2x1 행렬

# 편향: 뉴런을 흥분 시키는 정도, sigmoid를 사용시 그리드가 shift 됨
b_im = np.array([3.0, -3.0]) # 은닉층
b_mo = np.array([0.1]) # 출력층

# 은닉층
def middle_layer(x, w, b):
    u = np.dot(x, w) + b # 입력과 가중치 곱의 합 + 편향
    return activaton.sigmoid_function(u)

def output_layer(x, w, b):
    u = np.dot(x, w) + b # 입력과 가중치 곱의 합 + 편향
    return u

# 그리드맵의 각 그리드별 뉴런의 연산
for i in range(10):
    for j in range(10):

        inp = np.array([X[i], Y[j]])        # input layer
        mid = middle_layer(inp, w_im, b_im) # middle layer
        out = output_layer(mid, w_mo, b_mo) # output layer
        
        # 그리드맵에 출력값저장
        Z[j][i] = out[0]

plt.imshow(Z, "gray", vmin = 0.0, vmax = 1.0)
plt.colorbar()
plt.show()
        


