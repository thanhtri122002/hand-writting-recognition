from keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import joblib
import os

ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',required= True, help = "path to input the image")
ap.add_argument('-m','--model', type = str , required= True , help = "path of the model you want to use")
args = vars(ap.parse_args())

model = load_model(args['model'])

image = cv2.imread(args["image"])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray",gray)

blurred = cv2.GaussianBlur(gray , (5,5), 0)


edged = cv2.Canny(blurred, 30, 150)
#cv2.imshow('edged',edged)


cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


cnts = imutils.grab_contours(cnts)

#output the sorted contours and the corresponding bounding boxes
cnts = sort_contours(cnts , method= 'left-to-right')[0]

chars = []

for i,c in enumerate(cnts):
    (x , y , w , h ) = cv2.boundingRect(c)
    if(w >=5 and w <=150) and (h>= 15 and h <=120):
        region_of_interest = gray[y : y + h , x : x+ w]
        thresh = cv2.threshold(region_of_interest, 0 , 255 , cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #cv2.imshow(f'threshed {i}', thresh)
        (tH, tW) = thresh.shape
        print('thresh shape')
        print(thresh.shape)#(46, 25)
        if tW > tH:
            thresh = imutils.resize(thresh, width = 28)
        else:
            thresh = imutils.resize(thresh , height = 28)
        (tH, tW) = thresh.shape
        #print(thresh.shape)#(28, 15)
       
        dX = int(max(0,32 - tW) / 2.0)
        dY = int(max(0,32 - tH) / 2.0)
        """
        cv2.copyMakeBorder(thresh , top = dY, bottom = dY, left = dX , 
        right = dX, borderType = cv2.BORDER_CONSTANT , value = (0,0,0)) 
        adds a border of black color around the thresh image to make it of size 32x32 pixels.
        """
        padded = cv2.copyMakeBorder(thresh , top = dY, bottom = dY, left = dX , right = dX, borderType = cv2.BORDER_CONSTANT , value = (0,0,0))
        padded = cv2.resize(padded, (28,28))
        padded = padded.astype('float32') / 255.0
        padded = np.expand_dims(padded , axis = -1)
        chars.append((padded , (x,y,w,h)))
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32") #number of words ,28,28 => 

# OCR the characters using our handwriting recognition model
preds = model.predict(chars)#pred list of probilities

label_names = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
label_list = [l for l in label_names]

for (pred , (x,y,w,h)) in zip(preds, boxes):
    i = np.argmax(pred) # choose the index of the highest probilities
    prob = pred[i] #the values of the highest probility in pred list of probilities
    label = label_list[i] 
    cv2.rectangle(image, (x,y), (x + w , y + h), (0,255,0), 2)
    cv2.putText(image , label , (x -10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),2)
cv2.imshow("Image", image)
cv2.waitKey(0)


    