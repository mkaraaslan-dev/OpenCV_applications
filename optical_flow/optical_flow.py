
import cv2
import numpy as np






cap=cv2.VideoCapture("kav.m4v")

feature_params = dict( maxCorners = 100,
                      qualityLevel = 0.3,
                      minDistance = 7,
                      blockSize = 7 )

lk_params = dict( winSize = (15,15),
maxLevel = 15,
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color=np.random.randint(0,255,(100,3))

ret,old_frame=cap.read()
old_gray=cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
p0=cv2.goodFeaturesToTrack(old_gray,mask=None,**feature_params)


mask=np.ones_like(old_frame)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fbgr=cv2.createBackgroundSubtractorMOG2()

hsv = np.zeros_like(old_frame)
hsv[...,1] = 255

def dense(frame):
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(old_gray,frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*10/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    result= cv2.add(frame,rgb)
    return result
def draw(frame):
    fgmask=fbgr.apply(frame_gray)
    
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # blur=cv2.GaussianBlur(frame,(5,5),0)
    # ret,fgmask=cv2.threshold(fgmask, 0,256,cv2.THRESH_BINARY)
    
    contours,_=cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        
        if cv2.contourArea(cnt)>120:
            # M=cv2.moments(cnt)
    
            # cx = int(M['m10']/M['m00'])
            # cy = int(M['m01']/M['m00'])
    
            # center=(cx,cy)    
         
            rect = cv2.minAreaRect(cnt)
            box=cv2.boxPoints(rect)
            box=np.int0(box)
            frame=cv2.drawContours(frame,[box],0,(0,255,0),1)
            # cv2.putText(frame,'insan',center,cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,255))    
            cv2.imshow('fgmask',fgmask)



while True:
    ret,frame=cap.read()
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame=dense(frame)
    draw(frame)
    
        
            
    
    p1,st,err=cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,p0,None,**lk_params)
        
    good_new=p1[st==1]
    good_old=p0[st==1]
    
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b=new.ravel()
        c,d=old.ravel()
        mask=cv2.line(mask,(a,b),(c,d),color[i].tolist(),2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)    
            # mask=cv2.rectangle(mask,(a,b),(c,d),color[i].tolist(),2)
    
    img = cv2.add(frame,mask)
    res = np.hstack((mask,img))   
    cv2.imshow('res',frame)
    # cv2.imshow('Mask',mask)
    k = cv2.waitKey(25) & 0xff
    if k == 27:
        break
   
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()