import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 図示する関数
def func(x, y):
    return np.sin(x) * np.sin(y)

# メッシュを作成
x = np.arange(-5.0, 5.0, 0.1)
y = np.arange(-5.0, 5.0, 0.1)
X, Y = np.meshgrid(x, y)

# plot_surfaceで曲面プロット
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection' : '3d'})
ax.plot_surface(X, Y, func(X, Y), rstride=1, cstride=10, cmap='jet', alpha=0.4)
plt.show()