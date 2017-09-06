import cv2

img = cv2.imread("../images/person_292.png")
crop_img = img[200:400, 100:300]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)