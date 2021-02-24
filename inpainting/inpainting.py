# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:37:31 2020

@author: Mahmut
"""


import cv2
import numpy as np

img2=cv2.imread('goruntu.jpg')
img2=cv2.resize(img2, (400,400))



drawing=False
mode=True
ix,iy=-1,-1

def draw(event,x,y,flags,param):
    global ix,iy,drawing,mode
    
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing==True
        ix,iy=x,y
                
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.rectangle(img,(ix,iy),(x,y),(255,255,255),-1)
                cv2.rectangle(img2,(ix,iy),(x,y),(255,0,0),1)
            else:
                cv2.line(img,(ix,iy),(x,y),(255,255,255),5)
                cv2.line(img2,(ix,iy),(x,y),(255,0,0),5)
                
            
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            print(x)
            cv2.rectangle(img,(ix,iy),(x,y),(255,255,255),-1)
            cv2.rectangle(img2,(ix,iy),(x,y),(255,0,0),1)
        else:
            cv2.line(img,(ix,iy),(x,y),(255,255,255),5)
            cv2.line(img2,(ix,iy),(x,y),(255,0,0),5)
            # cv2.circle(img,(ix,iy),3,(0,0,255),-1)
            # cv2.circle(img2,(ix,iy),3,(0,0,255),-1)
            
            


cv2.namedWindow('İslem penceresi')
cv2.setMouseCallback('İslem penceresi',draw)
img = np.ones((400,400,3), np.uint8)

while(1):
    # cv2.putText(img2,'Duz cizgi cizmek isterseniz m harfine basiniz',(100,20),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,255,0))    
    cv2.setMouseCallback('İslem penceresi',draw)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # cv2.imshow('image',img)
    cv2.imshow('İslem penceresi',img2)
     
   
    dst = cv2.inpaint(img2,img,3,cv2.INPAINT_TELEA)
    dst2 = cv2.inpaint(img2,img,3,cv2.INPAINT_NS)
    
    cv2.putText(dst,'TELEA_METHOF',(25,50),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,255,0))
    cv2.putText(dst2,'NS_METHOD',(25,50),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,255,0))
    
    # cv2.imshow('TELEA_METHOD',dst)
    # cv2.imshow('NS_METHOD',dst2)
    img=img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    result=np.vstack((np.hstack((img,img2)),np.hstack((dst,dst2))))
    
    cv2.imshow('Result',result)           
    
    
    k = cv2.waitKey(1) & 0xFF    
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
cv2.destroyAllWindows()            
            