import numpy as np

#二乗和誤差(mean squared error)の関数
def mean_squared_error(y,t):
    return 0.5 * np.sum((y - t)**2)  #教師データとニューラルネットワークの出力の差の二乗の和が2乗和誤差の数値となる

t = np.array([0,0,1,0,0,0,0,0,0,0])
y = np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])

print(mean_squared_error(y,t))

y = np.array([0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0])

print(mean_squared_error(y,t))

#交差エントロピー誤差
def cross_entropy_error(y,t):
    delta = 1e-7                           #deltaを足すことでnp.log(0)の-infを回避する
    return -np.sum(t*np.log(y + delta))    #教師データの正解ラベルの数値に対応するニューラルネットワークの出力の自然対数が交差エントロピー誤差の数値となる

t = np.array([0,0,1,0,0,0,0,0,0,0])
y = np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])

print(cross_entropy_error(y,t))

y = np.array([0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0])

print(cross_entropy_error(y,t))