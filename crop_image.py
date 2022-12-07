import os
import numpy as np
import cv2

#reference: https://www.life2coding.com/crop-image-using-mouse-click-movement-python/

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping, uncropped,i2,picts
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
        refPoint = [(x_start, y_start), (x_end, y_end)]
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imwrite("Cropped/nopolyp_"+i2,roi)
            print(i2)
            picts+=1
            print(picts)
        uncropped = False
    elif event == cv2.EVENT_RBUTTONUP:
        cropping = False
        uncropped = False
count = 0
picts = 0
im_fold = 'Dataset/train_images'
for i in os.listdir(im_fold):
    print(count)
    count+=1
    filename = os.path.join(im_fold,i)
    if os.path.isfile("Cropped/nopolyp_"+i):
        continue
    i2 = i
    mask = cv2.imread('Dataset/train_masks/'+i2)
    cv2.imshow("mask",mask)
    cropping = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0
    image = cv2.imread(filename)
    oriImage = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)
    uncropped = True
    while uncropped:
        i = image.copy()
        if not cropping:
            cv2.imshow("image", image)
        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)
        cv2.waitKey(1)



#img = cv2.imread("Diceloss_30.png")
#rows,cols,_ = img.shape
#cv2.imshow("image",img)
#cv2.waitKey(0)


im_path = 'Dataset/train_images/'
op_path = 'Cropped/'
for i in os.listdir(im_path):        
   break