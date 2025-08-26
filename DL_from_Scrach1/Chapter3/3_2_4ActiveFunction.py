import numpy as np
import matplotlib.pyplot as plt

#ステップ関数
def step_function(x):
    y = x > 0
    return y.astype(int)

#シグモイド関数
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#ReLU関数
def relu(x):
    return np.maximum(0,x)

#グラフ描画
x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)
y3 = relu(x)
plt.plot(x, y1, linestyle ="--", label = "step")
plt.plot(x, y2, label = "sigmoid")
plt.plot(x, y3, linestyle =":", label = "relu")
plt.show()