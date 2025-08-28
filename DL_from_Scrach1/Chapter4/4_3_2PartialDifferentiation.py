import matplotlib.pyplot as plt
import numpy as np

# 三次元のグラフ出力をするのでそれ用の関数をインポート
from mpl_toolkits.mplot3d import Axes3D


# 変数x1とx2それぞれの2乗を足し合わせる関数
def funcPD(x1, x2):
    return x1**2 + x2**2


# (x1,x2)データを作成
x1 = np.arange(-3.0, 3.0, 0.1)
x2 = np.arange(-3.0, 3.0, 0.1)
# 格子点を作成
X1, X2 = np.meshgrid(x1, x2)

Z = funcPD(X1, X2)

# 2次元グラフの作成
fig = plt.figure()
# 3次元グラフに拡張
ax = fig.add_subplot(111, projection="3d")


# 軸ラベルを設定
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x)")

# なぜかうまく出力されない
ax.plot_wireframe(X1, X2, Z)
plt.show()

# 出力したいデータを見ても原因はわからず
print(X1.shape)
print(X1)
print(Z.shape)
print(Z)
