# importing the module
import cv2

# function to display the coordinates of
# of the points clicked on the image




def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
      #  line_1_point_x = int(278)
      #  line_1_point_y = int(423)
      #  line_2_point_x = int(912)
      #  line_2_point_y = int(397)
       # cv2.line(img, (line_1_point_x, line_1_point_y),
       #          (line_2_point_x, line_2_point_y), (0, 255, 0), thickness=4)

        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)


# driver function
if __name__ == "__main__":
    # vid = cv2.VideoCapture(stream,cv2.CAP_GSTREAMER)
    vid = cv2.VideoCapture('./output_record_2.avi')
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    success, image = vid.read()
    while True:
        if success:
            cv2.imwrite("coordinate_x_y.jpg", image)
            # reading the image
            img = cv2.imread('coordinate_x_y.jpg', 1)
            vid.release()
            break

    # displaying the image
    cv2.imshow('image', img)

    # setting mouse hadler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()
