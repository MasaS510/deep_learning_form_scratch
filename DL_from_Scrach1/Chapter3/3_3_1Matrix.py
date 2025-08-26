import numpy as np

#多次元配列
A = np.array([[1,2], [3,4], [5,6]]) #2次元の配列を作成
print(A)
print(np.ndim(A)) #Aの次元数を取得する
print(A.shape)    #Aの要素数を次元ごとに取得

#行列の積
B = np.array([[7,8,9],[10,11,12]])
print(B)
print(np.dot(A,B)) #AとBのdot積（ドット積）を求める