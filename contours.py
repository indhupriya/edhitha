import numpy as np
import cv2

im = cv2.imread('DSC_0869cropped.JPG')
print im.shape
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


for cnt in contours:
   print cv2.contourArea(cnt)
   if cv2.contourArea(cnt)>500 and cv2.contourArea(cnt)<2000:
       cv2.drawContours(im,[cnt],0,(0,255,0),3)
cv2.imshow("window title", im)
cv2.waitKey()