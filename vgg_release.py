import numpy as np,sys
from scipy.signal import convolve2d
import skimage.measure
import cv2
import glob
import matplotlib.pyplot as plt
import scipy
from scipy import misc
from scipy import ndimage as ndi
from skimage import feature
import json
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-g', '--graphical',action='store_const', const=True)
parser.add_argument('-s', '--save',action='store_const', const=True)
res = parser.parse_args()

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def ReLu(x):
    mask = (x>0) * 1.0
    return mask *x

def d_ReLu(x):
    mask = (x>0) * 1.0
    return mask

file = open ('./nn_weights')
js = json.load(file)

w1 = js['conv_1']
w2 = js['conv_2']
w3a = js['conv_3a']
w3b =js['conv_3b']
w4a = js['conv_4a']
w4b = js['conv_4b']
w5 = js['fc5']
w6 = js['fc6']
w7 = js['fc7']

res_0 = glob.glob("./dataset/paper/testset/0/*")
res_1 = glob.glob("./dataset/paper/testset/1/*")
images = [item for sublist in zip(res_0,res_1) for item in sublist]


def recognize(w1, w2, w3a, w3b, w4a, w4b, w5, w6, w7, img, res):
    _image_ = cv2.imread(img, 0)
    image = (feature.canny(_image_, sigma=1)).astype(float)
    image = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

    l1 = ReLu( convolve2d(np.pad(image,1,'constant'),w1,'valid') )
    l2 = ReLu( convolve2d(np.pad(l1,1,'constant'),w2,'valid') )

    l2_Mean = skimage.measure.block_reduce(l2, (2,2), np.mean)

    l3a = ReLu( convolve2d(np.pad(l2_Mean,1,'constant'),w3a,'valid') )
    l3b = ReLu( convolve2d(np.pad(l2_Mean,1,'constant'),w3b,'valid') )

    l4a = ReLu(  convolve2d(np.pad(l3a,1,'constant'),w4a,'valid') )
    l4b = ReLu( convolve2d(np.pad(l3b,1,'constant'),w4b,'valid') )

    l4a_Mean = skimage.measure.block_reduce(l4a, (2,2), np.mean)
    l4b_Mean = skimage.measure.block_reduce(l4b, (2,2), np.mean)

    _l5 = np.expand_dims(np.hstack(( l4a_Mean.ravel(), l4b_Mean.ravel() )),axis=0)
    l5 = ReLu( _l5.dot(w5) )
    l6 = ReLu( l5.dot(w6) )
    l7_Output = tanh( l6.dot(w7) )

    ig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 4),sharex=True, sharey=True)
    if round(float(l7_Output)) == 0:
        ax1.set_title('nn out: '+str(l7_Output)+" - smooth paper" , fontsize=20)
    if round(float(l7_Output)) == 1:
        ax1.set_title('nn out: '+str(l7_Output)+" - crumpled paper", fontsize=20)
    ax1.imshow(_image_, cmap=plt.cm.gray)

    if res.save:
        plt.savefig('./nn_out/'+str(i)+'.png')
    if res.graphical:
        plt.show()

    return l7_Output

ts_t = time.time()
i = 0
for img in images:
    ts = time.time()
    nn_out = recognize(w1, w2, w3a, w3b, w4a, w4b, w5, w6, w7, img, res)
    te = time.time()

    if i % 2 == 0:
        print("train_label: [0.0], nn_output: ",  nn_out, " time: ", te-ts, "s")
    if i % 2 == 1:
        print("train_label: [1.0], nn_output: ",  nn_out, " time: ", te-ts, "s")
    i+=1
te_t = time.time()

print("\nnumber of images: ", i)
print("\ntotal time: ", te_t-ts_t, "s")