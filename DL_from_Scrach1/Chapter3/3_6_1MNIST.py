import numpy as np

#パスの指定に利用するライブラリをインポート
import sys

#ファイルのパスを指定する
sys.path.append('C:\\Users\\mstk\\Python学習用フォルダ\\DeepLearning\\DL_from_Scrach1\\deep-learning-from-scratch-master')

#関数の読み込み
from DLdataset.mnist import load_mnist  #ライブラリにdatasetが存在しているため、そちらを参照しないように名前を"DLdataset"と変更
                                        #波線は上のファイルパス指定を実行していないために出ていると考えられる

# load_mnist関数を実行
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

#画像表示用モジュールをインポート
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_train[0]
label = t_train[0]

img = img.reshape(28,28)    #変数imgに格納されているデータの形状をもとの画像サイズに整形

#画像と答えの出力
print(label)
img_show(img)