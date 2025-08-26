import matplotlib.pyplot as plt
import numpy as np
#三次元のグラフ出力をするのでそれ用の関数をインポート
from mpl_toolkits.mplot3d import Axes3D 

def funcPD(x1,x2):
    return x1**2 + x2**2

X = np.arange(-3.0,3.0,0.1)

x1, x2 = np.meshgrid(X, X)

graph1 = funcPD(x1,x2)

#2次元グラフの作成
fig = plt.figure()
#3次元グラフに拡張
ax = Axes3D(fig)


# 軸ラベルを設定
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x)")

#なぜかうまく出力されたない
ax.plot_wireframe(x1,x2,funcPD(x1,x2))
plt.show()

#出力したいデータを見ても原因はわからず
print(x1)
print(graph1.shape)
print(graph1)
