import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

X = np.array([1.0, 0.5])                           #入力信号
W1 = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])   #第一層の重み
B1 = np.array([0.1, 0.2, 0.3])                     #第一層のバイアス

A1 = np.dot(X,W1) + B1                             #隠れ層(第一層)の重み付き和
print(A1)
Z1 =sigmoid(A1)                                    #活性化関数による信号への変換
print(Z1)

W2 =np.array([[0.1, 0.4],[0.2, 0.5],[0.3, 0.6]])  #第二層の重み
B2 = np.array([0.1, 0.2])                         #第二層のバイアス

A2 = np.dot(Z1,W2) + B2                             #隠れ層(第二層)の重み付き和
Z2 =sigmoid(A2)                                    #活性化関数による信号への変換

W3 = np.array([[0.1, 0.3],[0.2, 0.4]])             #第三層の重み
B3 = np.array([0.1, 0.2])                          #第三層のバイアス

def identity_function(X):                          #恒等関数の定義
    return X

A3 = np.dot(Z2,W3) + B3                            #重み付き和
Y = identity_function(A3)                          #出力層の信号

print(Y)
 