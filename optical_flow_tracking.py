# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 17:42:17 2020

@author: Mahmut
"""
import math
import cv2
import numpy as np

cap=cv2.VideoCapture(0)

#mause function
def selec_point(event,x,y,flags,params):
    # point verilerini daha sonra while dongusunde kullanmak için global yapıyoruz
    global point,point_selected,old_points
    # mausun sol tuşuna basıldığı andaki konumunu alıyoruz 
    if event==cv2.EVENT_LBUTTONDOWN:
        point=(x,y)
        # bir anahtar oluşturuyoruz
        point_selected=True
        old_points=np.array([[x,y]],dtype=np.float32)
# frame adında boş bir pencere açıyoruz
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', selec_point)        
# bir anahtar oluşturuyoruz
point_selected=False
point=()
old_points=np.array([[]])
# create old frame

_,frame=cap.read()
frame=cv2.resize(frame,(800,600))
old_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


lk_params = dict( winSize = (150,150),
maxLevel = 9,
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while (cap.isOpened()):
    _,frame=cap.read()
    frame=cv2.resize(frame,(800,600))
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if point_selected is True:
        # cv2.circle(frame, point, 5,(0,255,0),2)
        new_points,status,error=cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_points,None,**lk_params)
                
        # print(new_points)
        x,y=new_points.ravel()
        x1=float(x)
        y1=float(y)
        x2=float(750)
        y2=float(750)
        angle = int(math.atan((y1-y2)/(x2-x1))*180/math.pi)
        font=cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame,str(angle),(775,740),font,1,(255,255,250),2,cv2.LINE_AA)
        # print(angle)
        # cv2.line(frame,(750,750),(x,y),(255,0,0),5)
        cv2.line(frame,(750,750),(1500,750),(255,0,0),5)
        # cv2.circle(frame, (x,y),5,(0,0,255),-1)
        old_gray=frame_gray.copy()
        old_points=new_points
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
    
    
cap.release()
cv2.destroyAllWindows()    