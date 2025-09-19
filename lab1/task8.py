import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x, center_y = w//2, h//2

    b, g, r = frame[center_y, center_x]

    colors = {
        "red":   (0, 0, 255),
        "green": (0, 255, 0),
        "blue":  (255, 0, 0)
    }

    min_dist = float("inf")
    color = (0, 0, 0)
    for c in colors.values():
        dist = np.sqrt((int(b)-c[0])**2 + (int(g)-c[1])**2 + (int(r)-c[2])**2)
        if dist < min_dist:
            min_dist = dist
            color = c

    horiz_w = 160
    horiz_h = 25
    vert_w = 25
    vert_h = 200

    horiz_TPoint = (center_x-horiz_w//2, center_y-horiz_h//2)
    horiz_BPoint = (center_x+horiz_w//2, center_y+horiz_h//2)
    cv2.rectangle(frame, horiz_TPoint, horiz_BPoint, color, -1)

    vertTop_TPoint = (center_x-vert_w//2, center_y-vert_h//2)
    vertTop_BPoint = (center_x+vert_w//2, center_y-horiz_h//2)
    cv2.rectangle(frame, vertTop_TPoint, vertTop_BPoint, color, -1)

    horizBottom_TPoint = (center_x-vert_w//2, center_y+horiz_h//2)
    horizBottom_BPoint = (center_x+vert_w//2, center_y+vert_h//2)
    cv2.rectangle(frame, horizBottom_TPoint, horizBottom_BPoint, color, -1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
