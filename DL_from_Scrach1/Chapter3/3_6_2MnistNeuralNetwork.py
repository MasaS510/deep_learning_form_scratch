import numpy as np
from common.functions import sigmoid, softmax
import pickle

#MNISTデータを読み込む関数を定義
def read_data():
    import sys
    sys.path.append('DeepLearning\\DL_from_Scrach1\\deep-learning-from-scratch-master')
    from DLdataset.mnist import load_mnist
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

#ネットワークの学習済みパラメータを読み込む
def read_network():
    with open("DeepLearning\\DL_from_Scrach1\\deep-learning-from-scratch-master\\ch03\\sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    return network

#ネットワークの作成
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

#
x, t = read_data()
network = read_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) #最も確率の高い要素のインデックスを取得
    if p == t[i]:   #推定値が答えと一致している場合に増やす
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))