import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage as ndi
from scipy import misc
import skimage.measure
from skimage import feature
import cv2
import glob
import scipy
import json

np.random.seed(1024)

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


num_epoch = 10
learing_rate = 0.0001
learing_rate_conv = 0.00001
total_cost = 0
cost_array =[]

w1 = np.random.randn(3,3)*np.sqrt(2/3)
w2 = np.random.randn(3,3)*np.sqrt(2/3)

w3a = np.random.randn(3,3)*np.sqrt(2/3)
w3b = np.random.randn(3,3)*np.sqrt(2/3)

w4a = np.random.randn(3,3)*np.sqrt(2/3)
w4b = np.random.randn(3,3)*np.sqrt(2/3)  #для relu - *np.sqrt(2/w_size)

w5 = np.random.randn(2048,1024)*np.sqrt(2/2048)
w6 = np.random.randn(1024,512)*np.sqrt(2/1024)
w7 = np.random.randn(512,1)*np.sqrt(1/512)  # для tanh - *np.sqrt(1/w_size)


res_0 = glob.glob("./dataset/paper/trainset/0/*") # ровные листы
res_1 = glob.glob("./dataset/paper/trainset/1/*") # мятые листы
res = [item for sublist in zip(res_0,res_1) for item in sublist] # объединяем массивовы в один с чередованием элементов

for iter in range(num_epoch):
	i = 0
	print("Epoch: ", iter)
	for img in res:
		label = np.array([0.000001])
		if i % 2 == 0:
			label = np.array([0.000001])
		if i % 2 == 1:
			label = np.array([0.999999])
		i+=1

		image = cv2.imread(img, 0)
		image = (feature.canny(image, sigma=1)).astype(float)
		image = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
		#image = image-np.mean(image, axis = 0)

		l1 = convolve2d(np.pad(image,1,'constant'),w1,'valid')
		l1_a = ReLu(l1)
		l2 = convolve2d(np.pad(l1_a,1,'constant'),w2,'valid')
		l2_a = ReLu(l2)

		l2Mean = skimage.measure.block_reduce(l2_a, (2,2), np.mean)

		l3a = convolve2d(np.pad(l2Mean,1,'constant'),w3a,'valid')
		l3a_a = ReLu(l3a)
		l3b = convolve2d(np.pad(l2Mean,1,'constant'),w3b,'valid')
		l3b_b = ReLu(l3b)
		l4a = convolve2d(np.pad(l3a_a,1,'constant'),w4a,'valid')
		l4a_a = ReLu(l4a)
		l4b = convolve2d(np.pad(l3b_b,1,'constant'),w4b,'valid')
		l4b_b = ReLu(l4b)

		l4aMean = skimage.measure.block_reduce(l4a_a, (2,2), np.mean)
		l4bMean = skimage.measure.block_reduce(l4b_b, (2,2), np.mean)

		_l5 = np.expand_dims(np.hstack(( l4aMean.ravel(), l4bMean.ravel() )),axis=0)
		l5 = _l5.dot(w5)
		l5a = ReLu(l5)

		l6 = l5a.dot(w6)
		l6a = ReLu(l6)

		l7 = l6a.dot(w7)
		l7_Output = tanh(l7)

		lay_7_lost = l7_Output - label
		lay_7_delta = lay_7_lost * d_ReLu(l7)
		layer = l6a
		lay_7_grad = layer.T.dot(lay_7_delta)
		w7 = w7 - learing_rate * lay_7_grad

		lay_6_lost = (lay_7_delta).dot(w7.T)
		lay_6_delta = lay_6_lost * d_ReLu(l6)
		layer = l5a
		lay_6_grad = layer.T.dot(lay_6_delta)
		w6 = w6 - learing_rate * lay_6_grad

		lay_5_lost = (lay_6_delta).dot(w6.T)
		lay_5_delta = lay_5_lost * d_ReLu(l5)
		layer = _l5
		lay_5_grad = layer.T.dot(lay_5_delta)
		w5 = w5 - learing_rate * lay_5_grad

		lay_4a_lost = np.reshape((lay_5_delta).dot(w5.T)[:,:1024],(32,32)).repeat(2,axis=0).repeat(2,axis=1)
		lay_4a_delta = lay_4a_lost * d_ReLu(l4a)
		layer = l3a_a
		lay_4a_grad = np.rot90(convolve2d(np.pad(layer,1,'constant'),np.rot90(lay_4a_delta,2),'valid'),2)
		w4a = w4a - learing_rate_conv * lay_4a_grad

		lay_4b_lost = np.reshape((lay_5_delta).dot(w5.T)[:,1024:],(32,32)).repeat(2,axis=0).repeat(2,axis=1)
		lay_4b_delta = lay_4b_lost * d_ReLu(l4b)
		layer = l3b_b
		lay_4b_grad = np.rot90(convolve2d(np.pad(layer,1,'constant'),np.rot90(lay_4b_delta,2),'valid'),2)
		w4b = w4b - learing_rate_conv * lay_4b_grad

		lay_3a_lost = convolve2d(w4a,np.rot90( np.pad(lay_4a_delta,1,'constant'),2)    ,'valid')
		lay_3a_delta = lay_3a_lost * d_ReLu(l3a)
		layer = l2Mean
		lay_3a_grad =np.rot90(convolve2d(np.pad(layer,1,'constant'),np.rot90(lay_3a_delta,2),'valid'),2)
		w3a = w3a - learing_rate_conv * lay_3a_grad

		lay_3b_lost = convolve2d(w4b,np.rot90( np.pad(lay_4b_delta,1,'constant'),2)    ,'valid')
		lay_3b_delta = lay_3b_lost * d_ReLu(l3b)
		layer = l2Mean
		lay_3b_grad =np.rot90(convolve2d(np.pad(layer,1,'constant'),np.rot90(lay_3b_delta,2),'valid'),2)
		w3b = w3b - learing_rate_conv * lay_3b_grad

		lay_2_lost = (convolve2d(w3a,np.rot90( np.pad(lay_3a_delta,1,'constant'),2),'valid') + convolve2d(w3b,np.rot90( np.pad(lay_3b_delta,1,'constant'),2),'valid')).repeat(2,axis=0).repeat(2,axis=1)
		lay_2_delta = lay_2_lost * d_ReLu(l2)
		layer = l1_a
		lay_2_grad = np.rot90(convolve2d(np.pad(layer,1,'constant'),np.rot90(lay_2_delta,2),'valid'),2)
		w2 = w2 - learing_rate_conv * lay_2_grad

		lay_1_lost = convolve2d(w2,np.rot90( np.pad(lay_2_delta,1,'constant'),2),'valid')
		lay_1_delta = lay_1_lost * d_ReLu(l1)
		layer = image
		lay_1_grad = np.rot90(convolve2d(np.pad(layer,1,'constant'),np.rot90(lay_1_delta,2),'valid'),2)
		w1 = w1 - learing_rate_conv * lay_1_grad


weights = dict(conv_1=w1.tolist(), conv_2=w2.tolist(), conv_3a=w3a.tolist(), conv_3b=w3b.tolist(), \
				 conv_4a=w4a.tolist(), conv_4b=w4b.tolist(), fc5=w5.tolist(), fc6=w6.tolist(), fc7=w7.tolist())
import json
with open('./nn_weights', 'w') as file:
	json.dump(weights, file)
	file.close()