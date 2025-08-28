import numpy as np


# 勾配を求める関数
def numerical_gradient(f, x):  # fは関数、xはnumpyの配列
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):  # 配列xに格納されている各座標の偏微分を求める
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h  # 前方差分を求める
        fxh1 = f(x)  # f(x+h)、教本通りだけど遠回りじゃない？

        x[idx] = tmp_val - h  # 後方差分を求める
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 中心差分を求める

        x[idx] = tmp_val

    return grad


# 勾配降下法の関数
def GDmethod(
    f, init_x, lr=0.01, step_num=100
):  # fは関数、init_xは初期値、lrは学習率(learning rate)、step_numは更新回数
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad  # 座標を更新

    return x


# 最小化の対象となる関数
def function2(x):  # 最小化する関数の設定
    return x[0] ** 2 + x[1] ** 2


init_x = np.array([-3.0, 4.0])  # 初期値の設定

print(GDmethod(function2, init_x, 0.1, 100))
