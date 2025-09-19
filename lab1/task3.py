import cv2

cap = cv2.VideoCapture('Venice.mp4', cv2.CAP_ANY)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.resize(gray, (w//3, h//3))

    cv2.imshow('Frame', frame)
    cv2.imshow('Gray and small', result)
    if cv2.waitKey(30) & 0xFF == 27:
        break