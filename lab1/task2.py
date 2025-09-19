import cv2

images = ["nature.jpg", "butterfly.png", "monalisa.webp"]
flags_im = [cv2.IMREAD_REDUCED_COLOR_8, cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE]
flags_window = [cv2.WINDOW_AUTOSIZE, cv2.WINDOW_NORMAL, cv2.WINDOW_FULLSCREEN]
for i in range(len(images)):
    image = cv2.imread(images[i], flags_im[i])
    cv2.namedWindow(f"My image{i+1}", flags_window[i])
    cv2.imshow(f"My image{i+1}", image)

cv2.waitKey(0)
cv2.destroyAllWindows()