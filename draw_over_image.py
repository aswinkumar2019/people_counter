import numpy as np
import cv2 
from time import sleep

# Making The Blank Image
vidcap = cv2.VideoCapture('/home/aswin/Documents/people-counting-opencv/videos/Atm.mp4')
success,image = vidcap.read()

drawing = False
ix = 0
iy = 0
# Adding Function Attached To Mouse Callback
def draw(event,x,y,flags,params):
    global ix,iy,drawing
    # Left Mouse Button Down Pressed
    if(event==1):
        drawing = True
        ix = x
        iy = y
    if(event==0):
        if(drawing==True):
            #For Drawing Line
            print('Drawing from',ix,iy ,'to' ,x,y)
            cv2.line(image,pt1=(ix,iy),pt2=(x,y),color=(255,255,255),thickness=3)
            cv2.imshow("Window",image)
            ix = x
            iy = y
            # For Drawing Rectangle
            # cv2.rectangle(image,pt1=(ix,iy),pt2=(x,y),color=(255,255,255),thickness=3)
    if(event==4):
        drawing = False



# Making Window For The Image
cv2.namedWindow("Window")

# Adding Mouse CallBack Event
cv2.setMouseCallback("Window",draw)

# Starting The Loop So Image Can Be Shown
while(True):
    success,image = vidcap.read()    
    cv2.imshow("Window",image)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
