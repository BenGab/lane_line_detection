import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny_edge(img, low_tershold, high_tershold):
    return cv2.Canny(img, low_tershold, high_tershold)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, treshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, treshold, np.array([]), minLineLength = min_line_len, maxLineGap = max_line_gap)
    line_img = np.zeros(img.shape, dtype = np.uint8)
    draw_lines(line_img, line_img)
    return line_img

def separate_lines(lines):
    right = []
    left = []
    for x1,y1,x2,y2 in lines[:, 0]:
        m = (float(y2) - y1) / (x2 - x1)
        if m >= 0: 
            right.append([x1,y1,x2,y2,m])
        else:
            left.append([x1,y1,x2,y2,m])
    
    return right, left

def draw_lines(img, lines, color=(255, 0, 0), thickness=7):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def lane_detection_pipeline(img):
    ### Params for region of interest
    bot_left = [80, 540]
    bot_right = [980, 540]
    apex_right = [510, 315]
    apex_left = [450, 315]
    vertices = [np.array([bot_left, bot_right, apex_left, apex_right], dtype = np.int32)]
    gray = grayscale(img)
    blur = gaussian_blur(gray, 7)
    edge = canny_edge(img, 50, 125)
    masked = region_of_interest(edge, vertices)
    lines = cv2.HoughLinesP(masked, 0.8, np.pi/180, 25, np.array([]), minLineLength=50, maxLineGap=200)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return img

img = cv2.imread("/home/bennyg/Images/lane.jpg")
final_img = lane_detection_pipeline(img)

cv2.imshow("image", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()