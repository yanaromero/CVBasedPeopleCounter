# Standard imports
import cv2
import numpy as np;
import math
from math import hypot

# Read image
cam = cv2.VideoCapture('AkbayRedder.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX

def distanceBlob (blob1, blob2):    
    (x1,y1),radius1 = cv2.minEnclosingCircle(blob1)
    center1 = (int(x1),int(y1))
    radius1 = int(radius1)
    (x2,y2),radius2 = cv2.minEnclosingCircle(blob2)
    center2 = (int(x2),int(y2))
    radius2 = int(radius2)

    return (math.hypot(int(x2)-int(x1),int(y2)-int(y1)))-(radius1+radius2)


while(1):
        ret, frame = cam.read()

        if not ret:
            break

        canvas = frame.copy()


        lower = (0,0,150)  #130,150,80
        upper = (0,100,255) #250,250,120
        mask = cv2.inRange(frame, lower, upper)
        

        blur = cv2.GaussianBlur(frame,(5,5),0)
        areaArray = []


        try:
            # NB: using _ as the variable name for two of the outputs, as they're not used
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            blob = max(contours, key=lambda el: cv2.contourArea(el))
            M = cv2.moments(blob)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            (x,y),radius = cv2.minEnclosingCircle(blob)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(canvas,center,radius,(0,255,0),2)
            #cv2.circle(canvas, center, 2, (0,255,255), 200, 1)
            

            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                areaArray.append(area)
            

            sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
            if(len(sorteddata)>1):
                secondlargestcontour = sorteddata[1][1]
                N = cv2.moments(secondlargestcontour)
                ncenter = (int(N["m10"] / N["m00"]), int(N["m01"] / N["m00"]))
                dist = distanceBlob(blob, secondlargestcontour)
                cv2.putText(canvas,str(dist),(0,160), font, .5,(0,255,255),1,cv2.LINE_AA)
                (x1,y1),radius1 = cv2.minEnclosingCircle(secondlargestcontour)
                center1 = (int(x1),int(y1))
                radius1 = int(radius1)
                if(dist>0):
                    cv2.circle(canvas,center1,radius1,(0,255,0),2)






        except (ValueError, ZeroDivisionError):
            pass


        cv2.putText(canvas,str(len(contours)),(0,120), font, .5,(0,255,255),1,cv2.LINE_AA)

        cv2.imshow('frame',frame)
        #cv2.imshow('blur',blur)
        cv2.imshow('canvas',canvas)
        cv2.imshow('mask',mask)

        
        if cv2.waitKey(50) & 0xFF == ord('q'):
                break
#im.release()
cv2.destroyAllWindows()