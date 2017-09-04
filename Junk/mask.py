import cv2
import numpy as np
im = cv2.imread("roi.png")
height,width,depth = im.shape
circle_img = np.zeros((height,width), np.uint8)

#cv2.circle(circle_img,(width/2,height/2),63,1,thickness=-1)
cv2.ellipse(circle_img,(width/2,height/2),(48,64),0,0,360,255,thickness=-1)

masked_data = cv2.bitwise_and(im, im, mask=circle_img)

cv2.imshow("masked", masked_data)
cv2.waitKey(0)