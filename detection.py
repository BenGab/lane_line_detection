import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.imread("/home/bennyg/Images/lane.jpg")
gray = grayscale(img)

cv2.imshow("image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()