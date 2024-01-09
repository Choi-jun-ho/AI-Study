import numpy as np
import matplotlib.pyplot as plt;

def step_function(x):
    return np.where(x<=0, 0, 1) # condition:조건에 따라 x:true, y:false를 출력하는 함수

def sigmoid_function(x):
    return 1/(1+np.exp(-x))

def tanh_function(x): 
    return np.tanh(x)

def relu_function(x):
    return np.where(x <= 0, 0, x)

def leaky_relu_function(x):
    return np.where(x <=0, 0.01*x, x)

def identify_function(x): # 항등함수
    return x

def softmax_function(x):
    # return np.exp(x)/np.sum(np.exp(x)) # 오버플로우 오류 발생
    return np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x))) # 수정

if __name__ == "__main__":
    x = np.linspace(-5, 5) # 지정한 범위 내에서 균일한 간격으로 숫자를 생성
    y = softmax_function(x)

    print(y) # condition : softmax_output

    plt.plot(x, y)
    plt.show()