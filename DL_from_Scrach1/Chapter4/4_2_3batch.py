import sys
sys.path.append('C:\\Users\\mstk\\Python学習用フォルダ\\DeepLearning\\DL_from_Scrach1\\deep-learning-from-scratch-master')

import numpy as np
from DLdataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label = False)

print(x_train.shape) #データの形状を出力
print(t_train.shape) #データの形状を出力

train_size = x_train.shape[0] #訓練データの数を取得する
batch_size = 10               #ミニバッチのデータ数を指定
batch_mask = np.random.choice(train_size, batch_size) #0から60000未満の数をランダムに10個取り出す
x_batch = x_train[batch_mask] #batch_maskの番号の訓練データ（画像）を取り出す
t_batch = t_train[batch_mask] #batch_maskの番号の訓練データ（正解）を取り出す

#バッチ対応版の交差エントロピー誤差
def cross_entropy_error(y, t):
    if y.ndim == 1:     #ndimで次元数が1であるか確認
        t = t.reshape(1, t.size) #ミニバッチのデータ数が1の場合は1列の二次元配列として整形する
        y = y.reshape(1, y.size) #ミニバッチのデータ数が1の場合は1列の二次元配列として整形する
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size #arange(batch_size)で0からバッチサイズ未満までの連番を生成する。
                                                                            #y[x,t]でxとtがリストであればそれぞれのインデックスで対応した数値が代入される

