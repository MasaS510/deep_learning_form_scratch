import numpy as np 

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5]) #重み(w)とバイアス(b)の数値を変更すると
    b = -0.7                 #ANDゲートおよびORゲートの作成が可能
    y = np.sum(x*w) + b
    if y <= 0:
        return 0
    elif y > 0:
        return 1
    
print(AND(0,0))