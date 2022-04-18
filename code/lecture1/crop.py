import cv2
import numpy as np
global hoop_pos
hoop_pos = np.zeros((2,2),np.int)

def position(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hoop_pos[0,:] = x,y
    if event == cv2.EVENT_RBUTTONDOWN:
        hoop_pos[1,:] = x,y

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image',position)
cap = cv2.VideoCapture("../../q.mp4")
while(cap.isOpened()):
    ret, frame=cap.read()
    if ret:
        cv2.imshow("image",frame)
        k=cv2.waitKey(40)
        if (k&0xff==ord('q')):
            break
cv2.destroyAllWindows()
if cap.isOpened():
    ret, frame=cap.read()
    if ret==True:
        cv2.rectangle(frame,(hoopPos[0,0],hoopPos[0,1]),(hoopPos[1,0],hoopPos[1,1]),(0,255,0),1)
        cv2.imshow("video",frame)
        k=cv2.waitKey(0)
        cv2.destroyAllWindows()
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame1 = frame1[hoopPos[0,1]:hoopPos[1,1],hoopPos[0,0]:hoopPos[1,0]]
        filename_pic = "C:/Users/86133/Desktop/NBA/dir" +str(count)+".png"
        cv2.imwrite(filename_pic,frame1)
        count += 1
        if frame_GRAY[0,0] == 0:
            break
cv2.destroyAllWindows()