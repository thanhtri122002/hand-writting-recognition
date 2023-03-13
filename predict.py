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

chars = []

for c in cnts:
    (x , y , w , h ) = cv2.boundingRect(c)
    if(w >=5 and w <=150) and (h>= 15 and h <=120):
        region_of_interest = gray[y : y + h , x : x+ w]
        thresh = cv2.threshold(region_of_interest, 0 , 255 , cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape

        if tW > tH:
            thresh = imutils.resize(thresh, width = 28)
        else:
            thresh = imutils.resise(thresh , height = 28)
    (tH, tW) = thresh.shape
    dX = int(max(0,32 - tW) / 2.0)
    dY = int(max(0,32 - tH) / 2.0)
    padded = cv2.copyMakeBorder(thresh , top = dY, bottom = dY, left = dX , right = dX, borderType = cv2.BORDER_CONSTANT , value = (0,0,0))
    padded = cv2.resize(padded, (28,28))
    padded = padded.astype('float32') / 255.0
    padded = np.expand_dims(padded , axis = -1)
    chars.append((padded , (x,y,w,h)))
    


    