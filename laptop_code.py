# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi

# import the necessary packages
from math import sqrt
import pickle
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from time import sleep
import collections
from numpy.linalg import norm
import math

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-o", "--output", type=str,
    help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
    help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=10,
    help="# of skip frames between detections")
args = vars(ap.parse_args())
cross_x = 0
cross_y = 0
set_flag1 = 0
set_flag2 = 0
track_count = 0
center = [0,0]
obj_id = [0] * 1000
tracker_last = 0
tracker_count = 0
to_reset = []
reset_time = []
total_count = 0
frame_count = []
# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
fp = open("shared.pkl","rb")
shared = pickle.load(fp)
x_start = shared["startx"]
y_start = shared["starty"]
x_end = shared["endx"]
y_end = shared["endy"]
#in_value = input("")
#if(y_start < y _end):
#check_x = (x[-1] >= (x_start - 2)) and (x[-1] <= (x_end + 2))
#check_y = (y[-1] >= (y_start - 2)) and (y[-1] <= (y_end + 2))
#elif()
slope = ((y_start - y_end) / (x_start - x_end))
a = slope
#print(math.tan(a))
#sleep(5)
angle = math.tan(a)
#sleep(5)
b = -1
c = y_start - (slope * x_start)
p1 = np.array([x_start,x_end])
p2 = np.array([y_start,y_end])
if(shared["in_type"] == 'file'):
  loc = shared["video_input"]
# load our serialized model from disk
#print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, grab a reference to the webcam
if(shared["in_type"] == 'webcam'):
    #print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    #print("[INFO] opening video file...")
    vs = cv2.VideoCapture(loc)

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    frame = vs.read()
    frame = frame[1] if shared["in_type"] == 'file' else frame

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if shared["in_type"] == 'file' and frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    #frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        #print(H,W)
        #sleep(5)

    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (W, H), True)

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % args["skip_frames"] == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            #print(startX,startY,endX,endY)
            #center = ((startX+endX) / 2 ,(startY+endY) / 2 )
            #p3 = np.array(center)
            #y-y1 =m(x-x1)
            #y-y1=mx-mx1
            #y-y1-mx+mx1 must be zero
            #mx-y+y1-mx1 must be zero
            #a = slope,b = -1,c = y1-(slope * x1) which is of form ax + by + c = 0
            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2)

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)
    c1 = 0
    for z in frame_count:
        if(frame_count == total_count):
            #print(frame_count,total_count + 1)
            obj_id[c1] = 0
            #print('reset frame ',c1)
            #sleep(5)
            c1 = c1 + 1
    ##print('Next frame')
    # loop over the tracked objects
    total_count = total_count + 1
    #print('total_count',total_count)
    for (objectID, centroid) in objects.items():
        center = [0,0]
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            #print(to)
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            ##print(objectID)
            y = [c[1] for c in to.centroids]
            x = [c[0] for c in to.centroids]
            ##print(x)
            ##print(y)
            distance = (abs((a * x[-1]) + (b * y[-1]) + c)) / sqrt((a * a) + (b * b))
            col_l = (y_end - y_start) / (x_end - x_start)
            col_r = (y[-1] - y_start) / (x[-1] - x_start)
            ##print('col_l',col_l)
            ##print('col_r',col_r)
            collinear = col_l - col_r
            ##print('collinearity :',collinear)
            ##print(distance)
            ##print('distance',distance)
            ##print(x_range.count(x[-1]))
            ##print(y_range.count(y[-1]))
            ##print(track_count)
            ##sleep(1)
            if(abs(x_start-x_end)>abs(y_start-y_end)):
                 #print('The line is horizontal')
                 if(angle<0):
                      print('y end is greater than y start')
                      check_x = (x[-1] >= (x_start - 2)) and (x[-1] <= (x_end + 2))
                      check_y = (y[-1] <= (y_start + 2)) and (y[-1] >= (y_end - 2))
                      print('In first if statement')
                      #print(check_x,check_y)
                      #print('x:',x[-1])
                      #print(x_start,x_end)
                      #sleep(1)
                 else:
                      #print('y start is greater than y end')
                      print('In 1st else statement')
                      check_x = (x[-1] >= (x_start - 2)) and (x[-1] <= (x_end + 2))
                      check_y = (y[-1] >= (y_start - 2)) and (y[-1] <= (y_end + 2))
                      #sleep(1)
            else:
                if(angle<0):
                    #print('In 2nd If statement')
                    check_x = (x[-1] >= (x_start - 2)) and (x[-1] <= (x_end + 2))
                    check_y = (y[-1] >= (y_start - 2)) and (y[-1] <= (y_end + 2))
                    #sleep(1)
                    #print('In 1st if statement')
                    #print(check_x,check_y)
                    #print('x end is greater than x start')
                    #print('In if statement')
                    #print(check_x,check_y)
                    #print(y_start,y_end)
                else:
                    check_x = (x[-1] >= (x_start - 2)) and (x[-1] <= (x_end + 2))
                    check_y = (y[-1] <= (y_start + 2)) and (y[-1] >= (y_end - 2))
                    #print('x start is greater than x end')
                    #print('In else statement')
                    #print('In 1st else statement')
                    #sleep(1)
                    #print('The line is vertical')
                    #print(check_x,check_y)
            if(collinear <= 0.05 and collinear>=-0.05 and check_x and check_y):
                if(obj_id[objectID] != 0):
                    print('It is the same person')
                    #print(x[-1],y[-1])
                    #print(check_x,check_y)
                    #print('In if statement')
                    ##print(obj_id[objectID])
                    #sleep(1)
                else:
                    #print(x[-1],y[-1])
                    #print('x:',x_range.count(x[-1]))
                    #print('y',y_range.count(y[-1]))
                    cross_x = cross_x + 1
                    #print("Object has crossed the line")
                    #print(check_x,check_y)
                    to_reset.append(objectID)
                    frame_count.append(total_count + 15)
                    to_reset.append(objectID)
                    #print('reset triggered')
                    #sleep(1)
                    #print(startX,startY,endX,endY)
                    obj_id[objectID] = 1
                    #print('In else statement')
                    ##sleep(5)
            track_count = track_count + 1
            to.centroids.append(centroid)
            to.counted = True
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    tracker_last = 0
    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Crossed", cross_x),
        ("Status", status),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()

# stop the timer and display FPS information
fps.stop()
#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#print("cross_x",cross_x)
#print("cross_y",cross_y)
# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
    vs.stop()

# otherwise, release the video file pointer
else:
    vs.release()

# close any open windows
cv2.destroyAllWindows()