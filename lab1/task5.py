import cv2

imageHSV = cv2.imread("nature.jpg")
hsv = cv2.cvtColor(imageHSV, cv2.COLOR_BGR2HSV) 
cv2.namedWindow("Not HSV", cv2.WINDOW_NORMAL)
cv2.namedWindow("HSV", cv2.WINDOW_NORMAL)
cv2.imshow("Not HSV", imageHSV)
cv2.imshow("HSV", hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()