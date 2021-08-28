# Python program to illustrate 
# saving an operated video
  
# organize imports
import numpy as np
import cv2
  


stream = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), " \
        "width=(int)1280, height=(int)720,format=(string)NV12, " \
        "framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, " \
        "format=(string)BGRx ! videoconvert ! video/x-raw, " \
        "format=(string)BGR ! appsink"

stream2="nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), " \
        "width=(int)1280, height=(int)720,format=(string)NV12, " \
        "framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, " \
        "format=(string)BGRx ! videoconvert ! video/x-raw, " \
        "format=(string)BGR ! appsink"


# This will return video from the first webcam on your computer.
vid = cv2.VideoCapture(stream,cv2.CAP_GSTREAMER)
vid2 = cv2.VideoCapture(stream2,cv2.CAP_GSTREAMER)
fps = int(vid.get(cv2.CAP_PROP_FPS))
fps2 = int(vid2.get(cv2.CAP_PROP_FPS))
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./output_record_1.avi', fourcc, fps, (int(vid.get(3)), int(vid.get(4))))

fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
out2 = cv2.VideoWriter('./output_record_2.avi', fourcc2, fps2, ((int(vid2.get(3))), (int(vid2.get(4)))))
framecount =0
# loop runs if capturing has been initialized. 
while(True):
    print(framecount)
    framecount = framecount+1
    # reads frames from a camera 
    # ret checks return at each frame
    ret, frame = vid.read()
    ret2,frame2 =  vid2.read() 
  
    # Converts to HSV color space, OCV reads colors as BGR
    # frame is converted to hsv
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)


    # output the frame
    out.write(frame) 
    out2.write(frame2) 
    # The original input frame is shown in the window 
    # cv2.imshow('Camera 2', frame2)

  
    # The window showing the operated video stream 
    # cv2.imshow('Camera 1', frame)
  
      
    # Wait for 'a' key to stop the program 
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
  
# Close the window / Release webcam
vid.release()
vid2.release()
# After we release our webcam, we also release the output
out.release() 
out2.release() 
# De-allocate any associated memory usage 
cv2.destroyAllWindows()
