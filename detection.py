import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

img = cv2.imread("/home/bennyg/Images/lane.jpg")
gray = grayscale(img)
blur = gaussian_blur(gray, 7)

cv2.imshow("image", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()