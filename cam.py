import cv2
J=0
while True:
	c =  cv2.VideoCapture(0)
	r, i = c.read()
	print(r, " img № "+str(J))
	cv2.imwrite('./dataset/paper/testset/1/crumpled_'+str(J)+'.png', i)
	c.release()
	J+=1
	input("*pause*")
