import numpy as np

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)         #ネイピア数の指数関数を作成
sum_exp_a = np.sum(exp_a) #上で作成した指数関数の和

y = exp_a / sum_exp_a     #ソフトマックス関数の作成

print(y)

#以下は関数として定義したソフトマックス関数

def softmax(a):
    c = np.max(a)          #aの最大値をcとしてとり
    exp_a = np.exp(a - c)  #aからcを引くことで指数関数の数値を小さくしオバーフローの対策を行う　
    sum_exp_a = np.sum(exp_a) 
    y = exp_a / sum_exp_a   
    return y