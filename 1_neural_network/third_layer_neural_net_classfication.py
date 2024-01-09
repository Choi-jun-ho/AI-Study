import numpy as np
import matplotlib.pyplot as plt
import activaton

X = np.arange(-1.0, 1.0, 0.1) # 원소수는 20개
Y = np.arange(-1.0, 1.0, 0.1)

# 가중치: 뉴런의 흥분 조건 설정, 그리드 회전에 대한 영향력
w_im = np.array([[1.0, 2.0],
                 [2.0, 3.0]]) # 은닉층 2x2 행렬
w_mo = np.array([[-1.0, 1.0],
                 [1.0, -1.0]]) # 출력층 2x2 행렬

# 편향: 뉴런을 흥분 시키는 정도, sigmoid를 사용시 그리드가 shift 됨
b_im = np.array([0.3, -0.3]) # 은닉층
b_mo = np.array([0.4, 0.1]) # 출력층

# 은닉층
def middle_layer(x, w, b):
    u = np.dot(x, w) + b # 입력과 가중치 곱의 합 + 편향
    return activaton.sigmoid_function(u)

def output_layer(x, w, b):
    u = np.dot(x, w) + b # 입력과 가중치 곱의 합 + 편향
    return activaton.softmax_function(u)

x_1 = []
y_1 = []
x_2 = []
y_2 = []

# 그리드맵의 각 그리드별 뉴런의 연산
for i in range(10):
    for j in range(10):

        inp = np.array([X[i], Y[j]])        # input layer
        mid = middle_layer(inp, w_im, b_im) # middle layer
        out = output_layer(mid, w_mo, b_mo) # output layer
        
        # 그리드맵에 출력값저장
        if out[0] > out[1]:
            x_1.append(X[i])
            y_1.append(Y[j])
        else:
            x_2.append(X[i])
            y_2.append(Y[j])

plt.scatter(x_1, y_1, marker="+")
plt.scatter(x_2, y_2, marker="o")
plt.show()
        


