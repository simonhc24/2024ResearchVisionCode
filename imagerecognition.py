import cv2
import numpy as np

def sobel_edge_detector(img):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad * 255 / grad.max()).astype(np.uint8)
    cv2.imshow('Edges', grad_norm)
    cv2.waitKey(0)
    return grad_norm

def color_line_detector(img):
    lower = np.array([90,110,100] )
    upper = np.array([110,140,240])
    hsv =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lower, upper)
    cv2.imshow("blue", mask)
    cv2.waitKey()
    cv2.destroyAllWindows()

img = cv2.imread('CarletonLiquidLensImages\Screenshot 2024-04-25 124841.png') 
img_blur = cv2.GaussianBlur(img, (1,1), 0) 
cv2.imshow("original", img_blur)
cv2.waitKey(0)
# Convert to graycsale
img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
color_line_detector(img_blur)
#sobel_edge_detector(img_blur)