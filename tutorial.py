import cv2 as cv
import sys
import numpy as np

img = cv.imread(cv.samples.findFile("mainimage.jpg"))

if img is None:
    sys.exit("Could not read the image")

cv.line(img, (600,10),(1000,1000),(0,255,0),4) #Draw a line: pt1, pt2, color, thickness
cv.rectangle(img,(500,0),(900,60),(0,0,255),3) #Draw rectangle
cv.circle(img,(450,100), 100, (0,0,255),1)
cv.ellipse(img,(256,256),(100,50),90,0,360,255,1)

pts = np.array([[100,50],[200,300],[700,200],[500,100]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(0,255,255))

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,0),2,cv.LINE_AA)
BLUE = [255,0,0]
img = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)
print(img.shape)
cv.imshow("Display window", img)
k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("output.jpg",img)





# import numpy as np
# import cv2 as cv
# drawing = False # true if mouse is pressed
# mode = True # if True, draw rectangle. Press 'm' to toggle to curve
# ix,iy = -1,-1
# # mouse callback function
# def draw_circle(event,x,y,flags,param):
#     global ix,iy,drawing,mode
#     if event == cv.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix,iy = x,y
#     elif event == cv.EVENT_MOUSEMOVE:
#         if drawing == True:
#             if mode == True:
#                 cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#             else:
#                 cv.circle(img,(x,y),5,(0,0,255),-1)
#     elif event == cv.EVENT_LBUTTONUP:
#         drawing = False
#         if mode == True:
#             cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#         else:
#             cv.circle(img,(x,y),5,(0,0,255),-1)

# img = np.zeros((512,512,3), np.uint8)
# cv.namedWindow('image')
# cv.setMouseCallback('image',draw_circle)
# while(1):
#     cv.imshow('image',img)
#     k = cv.waitKey(1) & 0xFF
#     if k == ord('m'):
#         mode = not mode
#     elif k == 27:
#         break
# cv.destroyAllWindows()