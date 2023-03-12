from keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',required= True, help = "path to input the image")
ap.add_argument('-m','--model', type = str , required= True , help = "path of the model you want to use")
args = vars(ap.parse_args())

model = load_model(args['model'])

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray , (5,5), 0)

edged = cv2.Canny(blurred, 30, 150)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts , method= 'left-to-right')

for c in cnts:
    (x , y , w , h ) = cv2.boundingRect(c)
    