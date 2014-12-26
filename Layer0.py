import numpy as np
import cv2
import os
import Tkinter as tknt
import tkFileDialog as tk
import tkMessageBox
root = tknt.Tk()
root.withdraw()

tkMessageBox.showinfo("Hello!", " Select your working directory carefully. Your working directory is a single folder with all your images and thou shall be blesseth")
imageDumpPath = tk.askdirectory()
tkMessageBox.showinfo("Next step!", " Press 'N' to move to the next image and if you think you found a target in the image ( YAY?), please remember to hit 'T' to see the masked image. \n P.S This is not case sensitive, so press away!")


def extractBlobs(image,origImage):	# UNFINISHED function that extracts the blob

	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to a single channel grayscale image
	image[image[:]>90]=255	# Neglect pixles with intensity less than 90
	mask = np.array(image,dtype = np.uint8)	# Create a mast using the backProjected image
	bit = cv2.bitwise_and(origImage,origImage,mask = mask)	# bitwise and the mask and the orignal image to regain clarity
	cv2.imwrite('./BlobbedImage.JPG',bit)
	cv2.imshow('im',bit)

def verifyTargetImage(image): # Function to carry out histogram backprojection and morphological operations(erosion) to create a mask which consists of the target and other noise.

	hsvImage = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)	# Converting the RGB image to the HSV color space

	hsvHistogram = cv2.calcHist([hsvImage],[0, 1], None, [180, 256], [0, 180, 0, 256] ) #   Calculating the histogram for channel 0(Hue) and channel 1(Saturation),
																						#	using a range of 0 to 180 for hue and 0 to 255 for saturation

	dst = cv2.calcBackProject([hsvImage],[0,1],hsvHistogram,[0,180,0,256],1)	#Filtering out the respective channels on the image using the given ranges
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
	cv2.filter2D(dst,5,disc,dst)
	ret,thresh = cv2.threshold(dst,150,255,1)
	thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
	res = cv2.bitwise_and(thresh,image)
	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	origImage = res.copy()
	res=cv2.erode(res,kernel,iterations=2)	# Eroding away all those tiny annoying blobs of grass
	extractBlobs(res,origImage)	# Function call, move up!
	cv2.waitKey(0)

if __name__=="__main__":

	for i,img in enumerate([os.path.join(imageDumpPath,fn) for fn in next(os.walk(imageDumpPath))[2]]):
		img = cv2.imread(img)
		try:
			img = cv2.resize(img,None, fx = 0.25, fy = 0.25)
		except:
			tkMessageBox.showinfo("The end?", " Either you've viewed all images in this folder or you selected the wrong folder. \n In case of the latter, run it again and select the right folder. Okay?")
			break
		cv2.imshow('image',img)
		k = cv2.waitKey(0)
		if k == ord('t'):	# Press 't' if an image has a target in it
			verifyTargetImage(img)
