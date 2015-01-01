import cv2
import numpy as np
def extractblob(im):
	print im.shape
	imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray,0,255,0)
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	z=0;
	for cnt in contours:
   		print cv2.contourArea(cnt)
                if cv2.contourArea(cnt)>1000 and cv2.contourArea(cnt)<3000:
     			cv2.drawContours(im,[cnt],0,(0,255,0),3)
     			x,y,w,h = cv2.boundingRect(cnt)
     			cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
     			crop=im[y:y+h,x:x+w]
     			cv2.imwrite("crop%d.jpg"%(z),crop)
     			cv2.imshow("crop%d"%(z),crop);
     			z=z+1;
        cv2.imwrite("windowtitled.JPG", im)
	cv2.imshow("hello",im)
	cv2.waitKey(0)

def backproject():
  im = cv2.imread("/home/sony/DSC_0869.JPG")
	roi = cv2.imread('DSC_0869cropped.JPG') 
	roiHsv = cv2.cvtColor( roi, cv2.COLOR_BGR2HSV)
	roiHist = cv2.calcHist([roiHsv],[0, 1], None, [256, 256], [0, 256, 0, 256] )
	hsv = cv2.cvtColor( im, cv2.COLOR_BGR2HSV)
	bckP = cv2.calcBackProject([hsv], [0,1], roiHist,[0,256,0,256], 1)
	ret,bckP = cv2.threshold(bckP,0,255,cv2.THRESH_BINARY_INV)
	bckM = cv2.merge(( bckP, bckP, bckP)) 
	mask = cv2.inRange(hsv, np.array([50,100,100], dtype = np.uint8), np.array([70,255,255], dtype = np.uint8))
	mask_inv = cv2.bitwise_not(mask)
	finalBP =  cv2.bitwise_and(im, bckM, mask = mask_inv)
	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	res=cv2.erode(finalBP,kernel,iterations=1)
	extractblob(res)

	
if __name__ == '__main__':
	 backproject()
	
