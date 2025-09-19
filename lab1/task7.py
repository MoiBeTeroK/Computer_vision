import cv2

def process_video(input_file, output_file):
    cap = cv2.VideoCapture(input_file)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_file, fourcc, 25, (width, height), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray)

        cv2.imshow('Video', gray)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

process_video(0, 'output_task7.mp4')