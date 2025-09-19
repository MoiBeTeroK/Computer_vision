import cv2
import numpy as np

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(frame_hsv, red_lower1, red_upper1) + cv2.inRange(frame_hsv, red_lower2, red_upper2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 0:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Размер объекта по контуру
            x, y, w, h = cv2.boundingRect(cnt)

            # Координаты черного прямоугольника по центру
            top_left = (cX-w//2, cY-h//2)
            bottom_right = (cX+w//2, cY+h//2)

            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), 2)
            cv2.circle(frame, (cX, cY), 3, (255, 0, 0), -1)

    cv2.imshow("Red_object_center", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
