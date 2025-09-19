import cv2

video = cv2.VideoCapture("http://192.168.1.128:8080/video")

while True:
    ok, frame = video.read()
    if not ok:
        break

    cv2.imshow("Camera_myPhone", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()

