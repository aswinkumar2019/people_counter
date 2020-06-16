import pickle
import numpy as np
import cv2 
from time import sleep
import imutils
# Making The Blank Image
input_type = input("Enter the type of input,webcam or file")
fp = open("shared.pkl","wb")
if(input_type == 'file'):
    location = input("Enter the location of the video file relative to the location of this shell script")
    vidcap = cv2.VideoCapture(location)
else:
    vidcap = cv2.VideoCapture(0)
success,image = vidcap.read()
height, width = image.shape[:2]
print(height,width)
#sleep(5)
drawing = False
ix = -100
iy = -100
x2 = -100
y2 = -100
# Adding Function Attached To Mouse Callback
def draw(event,x,y,flags,params):
    global ix,iy,drawing
    # Left Mouse Button Down Pressed
    if(event==1):
      if(ix > -50):
       print('You have chosen line between ',ix,iy, 'to', x,y)
       print('Set coordinates as ','(',ix,',',iy,')', ' and (',x,',',y,')')
       height, width = image.shape[:2]
       #sleep(5)
       if(input_type == "file"):
            shared = {"startx":ix, "starty":iy,"endx":x, "endy":y,"video_input":location,"in_type":input_type}
            pickle.dump(shared, fp)
            print(shared)
       else:
           shared = {"startx":ix, "starty":iy,"endx":x, "endy":y,"in_type":input_type}
           pickle.dump(shared, fp)
           print(shared)
       exit()
       x2 = x
       y2 = y
      else: 
        drawing = True
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
