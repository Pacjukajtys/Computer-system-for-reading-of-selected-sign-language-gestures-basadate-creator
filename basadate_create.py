import cv2
import numpy as np
import math
import csv
from PIL import Image

cap = cv2.VideoCapture(0)

while(1):
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi=frame[100:250, 100:250]
         
        cv2.rectangle(frame,(100,100),(250,250),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
         
    # define range of skin color in HSV
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
        
     #extract skin colur imagw  
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
    #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
    #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100) 
        
        font = cv2.FONT_HERSHEY_SIMPLEX

         
        #show the windows
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
    except:
        pass
    
    k = cv2.waitKey(5) & 0xFF
    


    
    #how letter is downlad
    
############################################ Basa Date    
    if k == 32:
        
        cv2.imwrite('test.jpg',mask)
        img=Image.open("C:/Users/Norbert/Desktop/MGR_FINAL_VERSION/tworzenie bazy danych do uczenia_high_rectangle/test.jpg")
        imgarray=np.array(img)
        
        plik=open("test.txt",'a')
        plik.write("1")
        for i in range(0, 150):
            for j in range (0, 150):
                plik.write(","+str(imgarray[i][j]))
        plik.write(",")
        plik.write("\n")
        plik.flush()
        plik.close 
############################################    
    
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()    
    