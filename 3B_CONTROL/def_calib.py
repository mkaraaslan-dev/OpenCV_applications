import cv2
import numpy as np
import glob
import time

cap=cv2.VideoCapture(0)

# take first frame of the video
frame =cv2.imread('hand.png')
# setup initial location of window

r,h,c,w = 250,200,250,200 # simply hardcoded the values
track_window = (c,r,w,h)
# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


# Load previously saved data
with np.load('C:\\Users\\Mahmut\\Desktop\\da\\B.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
# axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)





# img=cap.read()
img = cv2.imread('chessboard.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

if ret == True:
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
    _,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)



    imgpts = np.int32(imgpts).reshape(-1,2)

    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
        
        a=np.vstack((imgpts[0],imgpts[4],imgpts[5],imgpts[1]))    
        b=np.vstack((imgpts[1],imgpts[5],imgpts[6],imgpts[2]))    
        c=np.vstack((imgpts[2],imgpts[6],imgpts[7],imgpts[3]))    
        d=np.vstack((imgpts[3],imgpts[7],imgpts[4],imgpts[0]))
        t=imgpts[:4]
        l=imgpts[4:]
    
       
    
    
def draw(frame,ping,a,b,c,d,l,t):
    
    if ping==1:
        x=l.copy()
        l=d.copy()
        d=t.copy()
        t=b.copy()
        b=x.copy()
    if ping==2:
        x=b.copy()
        b=t.copy()
        t=d.copy()
        d=l.copy()
        l=x.copy()
    if ping==3:
        x=d.copy()
        d=a.copy()
        a=b.copy()
        b=c.copy()
        c=x.copy()
    if ping==4:
        x=c.copy()
        c=b.copy()
        b=a.copy()
        a=d.copy()
        d=x.copy()
    
        
   
   
    img = cv2.drawContours(frame,[a],-1,(255,255,0),-1)
    # img = cv2.drawContours(img,[b],-1,(0,255,255),-1)
    img = cv2.drawContours(frame, [t],-1,(0,255,0),-3)
    img = cv2.drawContours(frame,[c],-1,(0,255,255),-1)
    img = cv2.drawContours(frame,[d],-1,(255,0,255),-1)    
    img = cv2.drawContours(frame, [l],-1,(0,0,255),-1)  
    return a,b,c,d,l,t    

ping=0

while True:
    res,frame=cap.read()
     
    cv2.line(frame,(50,0),(0,4000),(0,255,0),2)
    cv2.line(frame,(500,0),(0,50000),(0,255,0),2)
    cv2.line(frame,(0,50),(7000,0),(0,255,0),2)
    cv2.line(frame,(0,450),(50000,0),(0,255,0),2)
    if res == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window        
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        
        cv2.imshow('img2',img2)
        if y<50:                           
            ping=1
            # print(y)
            time.sleep(0.5)
        if y>250:
            ping=2
            time.sleep(0.5)
        if x>275:
            ping=3
            time.sleep(0.5)
        if x<50:
            ping=4
            time.sleep(0.5)
        
        a,b,c,d,l,t= draw(frame,ping,a,b,c,d,l,t)
    say=0
    cv2.imshow('img',frame)
    
    ping=0
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
            # cv2.imwrite(fname[:6]+'.png', img)
cap.release()
cv2.destroyAllWindows()