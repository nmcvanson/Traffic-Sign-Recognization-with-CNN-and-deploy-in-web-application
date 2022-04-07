import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import imutils
import argparse
import os
import math
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

SIGNS = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Vehicle > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing vehicle > 3.5 tons'
            }
def predict_sign(img):
    model = load_model('./model/TSR.h5')
    data=[]
    #image = Image.open(img)
    image = cv2.resize(img, (30, 30))
    
    data.append(np.array(image))
    X_test=np.array(data)
    predict_x = model.predict(X_test) 
    Y_pred = np.argmax(predict_x,axis=1)
    #Y_pred = model.predict_classes(X_test)
    return Y_pred

def detect_red(img):
    imgBlur = cv2.GaussianBlur(img, (7,7), 1)
    hsv = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,50,50])
    upper2 = np.array([180,255,255])
    
    lower_mask = cv2.inRange(hsv, lower1, upper1)
    upper_mask = cv2.inRange(hsv, lower2, upper2)
    red_mask = lower_mask + upper_mask;

   
    
    kernal = np.ones((5, 5), "uint8")
	
	# For red color 
    red_mask = cv2.dilate(red_mask, kernal) 
    res_red = cv2.bitwise_and(img, img, mask = red_mask)
    return red_mask

class Video(object):
    SIGNS = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Vehicle > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing vehicle > 3.5 tons'
            }
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def predict_sign(img):
        model = load_model('./model/TSR.h5')
        data=[]
        #image = Image.open(img)
        image = cv2.resize(img, (30, 30))
        
        data.append(np.array(image))
        X_test=np.array(data)
        predict_x = model.predict(X_test) 
        Y_pred = np.argmax(predict_x,axis=1)
        #Y_pred = model.predict_classes(X_test)
        return Y_pred

    def detect_red(img):
        imgBlur = cv2.GaussianBlur(img, (7,7), 1)
        hsv = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,50,50])
        upper2 = np.array([180,255,255])
        
        lower_mask = cv2.inRange(hsv, lower1, upper1)
        upper_mask = cv2.inRange(hsv, lower2, upper2)
        red_mask = lower_mask + upper_mask;

    
        
        kernal = np.ones((5, 5), "uint8")
        
        # For red color 
        red_mask = cv2.dilate(red_mask, kernal) 
        res_red = cv2.bitwise_and(img, img, mask = red_mask)
        return red_mask
    def get_frame(self):
        ret,frame=self.video.read()
        copy_img = frame.copy()
        imgBlur = cv2.GaussianBlur(frame, (7,7), 1)
        hsv = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)
        #current_sign = None
        red_mask = detect_red(frame)
        

        
        #binary_image = preprocess_image(red_img)
        #binary_image = removeSmallComponents(binary_image, min_size_components)
        
        _, cnts, _= cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area > 2000:

                #cv2.drawContours(frame,[c],-1,(0,255,0), 1)

                M = cv2.moments(c)

                cx = int(M["m10"]/ M["m00"])
                cy = int(M["m01"]/ M["m00"])

                cv2.circle(frame,(cx,cy),7,(255,255,255),-1)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02*peri, True)
                x, y, w , h = cv2.boundingRect(approx)
                
                cropped_image = copy_img[y:y+h, x:x+w]
                Y_pred = predict_sign(cropped_image)
                s = [str(i) for i in Y_pred]
                sign_type = int("".join(s))
                text = SIGNS[sign_type]
                cv2.putText(frame,text,(20, 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0),2)
                cv2.imshow("sign",cropped_image)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()