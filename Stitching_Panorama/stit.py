import cv2
import os
import numpy as np
mainFolder='DataSet'
myFolders=os.listdir(mainFolder)

print(myFolders)


for folder in myFolders:
    path=mainFolder+'/'+folder
    images=[]
    myList=os.listdir(path)
    print(myList)
    print(f'total no of images detected {len(myList)}')
    
    for imgN in myList:
        curImg=cv2.imread(f'{path}/{imgN}')
        # curImg=cv2.resize(curImg,(0,0),None,0.2,0.2)
        images.append(curImg)
    # print(len(images))
    stitcher=cv2.Stitcher_create()        
   
    status,result=stitcher.stitch(images)
    # print(result)
    if (status==cv2.STITCHER_OK):
        
        # res=np.vstack((np.hstack((img,img2)),np.hstack((dst,dst2))))
        # a=5
        # for i in images:
        #     a=a+1
        #     cv2.imshow(str(a),i)
        cv2.imshow(folder,result)
        cv2.imwrite('Result.jpg',result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('Panorama Generated')
        
    else:
        print('Panorama Generation Unsuccessful')

