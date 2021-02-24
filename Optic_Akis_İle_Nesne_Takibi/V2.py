# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 22:52:11 2020

@author: Mahmut
"""

import math
import cv2
import numpy as np
from datetime import datetime
# import time

cap=cv2.VideoCapture(0)


_,frame=cap.read()
frame=cv2.resize(frame,(400,400))
old_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


lk_params = dict( winSize = (150,150),
maxLevel = 40,
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

old_points=np.array([[]])

switch=0
# datetime.tzinfo
start_time=datetime.now()
counter=0
while (cap.isOpened()):
    counter+=1
    _,frame=cap.read()
    frame=cv2.resize(frame,(400,400))
    # Convert BGR to HSV
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    dtime=datetime.now().ctime()
    # print(dtime)    
    
    font=cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(frame, str(dtime), (0, 19),font, 0.5, (255, 0,0), 1)        
    cv2.putText(frame, f"FPS:{counter / (datetime.now() - start_time).total_seconds()}", (0, 40),font, 0.5, (255, 0,0), 1)
    
    
            # define range of Red color in HSV
    lower = np.array([152,100, 150])
    upper= np.array([255,255,255])
            
            # Threshold the HSV image to get only red colors
    mask=cv2.inRange(hsv,lower,upper)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # Bitwise-AND mask and original image
    res=cv2.bitwise_and(frame,frame,mask=mask)
    for cnt in contours:
        if cv2.contourArea(cnt)>100:
            M=cv2.moments(cnt)
        
            cx = float(M['m10']/M['m00'])
            cy = float(M['m01']/M['m00'])
                
            center=(cx,cy)
            
            rect = cv2.minAreaRect(cnt)
            box=cv2.boxPoints(rect)
            box=np.int0(box)
            res=cv2.drawContours(res,[box],0,(0,255,0),3)
            # cv2.circle(frame, center,5,(0,0,255),-1)
            old_points=np.array([[cx,cy]],dtype=np.float32)
            # print(center)                
            switch=1
            # cv2.line(frame,(200,200),center,(255,0,0),5)
        # cv2.imshow('mask',mask)
        # cv2.imshow('res',res)
    # print(type(center))
    if (switch==1):    
        new_points,status,error=cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_points,None,**lk_params)
        x,y=new_points.ravel()
        if (error>19):    
            switch=0
        cv2.line(frame,(200,200),(x,y),(0,255,0),3)
        cv2.circle(frame, (x,y),5,(0,0,255),-1)    
        # print('error=',error)
        old_points=new_points
        old_gray=frame_gray.copy()
        x1=float(x)
        y1=float(y)
        x2=float(200)
        y2=float(200)
        angle = int(math.atan((y1-y2)/(x2-x1))*180/math.pi)
        # font=cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame,f"location:{x,y}", (0, 60),font, 0.5, (0, 255,255), 1)    
        cv2.putText(frame,f"angle:{angle}*",(0,75),font,0.5,(0,255,250),1)
        # print(new_points)
    # print(cx,cy)
    result= np.hstack((frame,res))
    # cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('result',result)
    # cv2.imshow('result',result)
    
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
  
cv2.destroyAllWindows() 
cap.release()    