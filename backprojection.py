import cv2
import numpy as np

#inverting image colours
def imInv ( imIn, inv):
	imIn = (255-imIn)
	cv2.imwrite(inv, imIn)

if __name__ == '__main__':

	im = cv2.imread("/home/praneeta/Downloads/DSC_0869.JPG")
	im = cv2.resize(im, None, fx = 0.25, fy = 0.25)
#	roi = im[200:300, 300:400] #random crop
	roi = cv2.imread('/home/praneeta/Downloads/DSC_0869cropped.JPG') #proper random crop
	
	#roi => hsv, histogram, normalize
	roiHsv = cv2.cvtColor( roi, cv2.COLOR_BGR2HSV)
	roiHist = cv2.calcHist([roiHsv],[0, 1], None, [256, 256], [0, 256, 0, 256] )
#	cv2.normalize(roiHist,roiHist,0,255, cv2.NORM_MINMAX)
	
	#image => hsv
	hsv = cv2.cvtColor( im, cv2.COLOR_BGR2HSV)

	bckP = cv2.calcBackProject([hsv], [0,1], roiHist,[0,256,0,256], 1)
#	bckP = cv2.medianBlur(bckP,5)
	ret,bckP = cv2.threshold(bckP,0,255,cv2.THRESH_BINARY_INV)
	bckM = cv2.merge(( bckP, bckP, bckP)) #make vector to array
#	imInv ( bckM, 'myBP0.jpg') #invert colours of back proj
#	bckM = (255-bckM)
#	finalBP =  cv2.bitwise_and(hsv, 'myBP0.jpg')
	mask = cv2.inRange(hsv, np.array([50,100,100], dtype = np.uint8), np.array([70,255,255], dtype = np.uint8))
	mask_inv = cv2.bitwise_not(mask)
	finalBP =  cv2.bitwise_and(im, bckM, mask = mask_inv)

#	cv2.imshow("back projection", bckP)
#	cv2.imshow('anded', finalBP)
	cv2.imwrite('myBP1.jpg', bckP)
	cv2.imwrite('myBP2.jpg', finalBP)
	cv2.waitKey(0)

