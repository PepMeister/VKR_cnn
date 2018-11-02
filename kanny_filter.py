import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage as ndi

from skimage import feature

import glob
res_0 = glob.glob("./dataset/paper/testset/0/*")
res_1 = glob.glob("./dataset/paper/testset/1/*")
res = [item for sublist in zip(res_0,res_1) for item in sublist]

i = 0
for img in res:
	img_array = cv2.imread(img, 0)
	edges1 = (feature.canny(img_array, sigma=1)).astype(float)
	edges2 = (feature.canny(img_array, sigma=3)).astype(float)

	fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

	ax1.imshow(img_array, cmap=plt.cm.gray)
	ax1.axis('off')
	if i % 2 == 0:
		ax1.set_title('smooth paper', fontsize=20)
	if i % 2 == 1:
		ax1.set_title('crumpled paper', fontsize=20)

	ax2.imshow(edges1, cmap=plt.cm.gray)
	ax2.axis('off')
	ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

	ax3.imshow(edges2, cmap=plt.cm.gray)
	ax3.axis('off')
	ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

	fig.tight_layout()
	plt.show()
	i+=1
