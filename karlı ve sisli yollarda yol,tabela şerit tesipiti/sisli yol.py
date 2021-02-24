# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 22:42:53 2020

@author: Mahmut
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import match_template


template=cv2.imread('sing.png',0)
template2=cv2.imread('sing2.png',0)

def rightdetection(frame,CLAHE):
    w, h= template.shape[::-1]    
    res=match_template(CLAHE,template)
    threshold=0.8
    # threshold=0.7
    loc=np.where(res>=threshold)    
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)
        font=cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame,'sagyap',(250,250),font,1,(0,0,255),0,cv2.LINE_AA)
    
def leftdetection(frame):    
    w, h= template2.shape[::-1]
    
    res=cv2.matchTemplate(gray,template2,cv2.TM_CCOEFF_NORMED)    
    threshold=0.8
    loc=np.where(res>=threshold)
    
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)
        font=cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame,'solyap',(250,250),font,1,(0,0,255),2,cv2.LINE_AA)
    
def stripdetection(frame):    
    edges = cv2.Canny(frame, 50, 200, None, 3)     
    minLineLength = 25
    maxLineGap = 150    
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, minLineLength,maxLineGap)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0,255,255),4, cv2.LINE_AA)
    
   
    
def drawroad(th,frame):
    contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
    for cnt in contours:
    
        epsilon=0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        frame=cv2.drawContours(frame,[approx],0,(0,255,0),-1)    
    
cap=cv2.VideoCapture('Sisli yollarda.mp4')

while True:
    _,frame=cap.read()
    
    
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray, (600, 300))
    frame=cv2.resize(frame, (600, 300))
    result=frame.copy()
    original=frame.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
        
    rest,thresh=cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
 
 
    
    # stripdetection(frame)        
    rightdetection(frame,cl1)
    rightdetection(frame,cl1)
    
    # stripdetection(result)
    # drawroad(thresh,result)
    rightdetection(result,cl1)
    leftdetection(result) 
    
       
        
    
    
    cv2.imshow('original_image',original)
    cv2.imshow('thresh',thresh)
    cv2.imshow('frame',frame)
    cv2.imshow('CLAHE(Kontrast Sınırlı Uyarlanabilir Histogram Eşitleme)',cl1)
    cv2.imshow('result',result)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
