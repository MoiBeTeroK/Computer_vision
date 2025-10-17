import cv2
import numpy as np
import math

def gaussian_kernel(n, sigma):
    a = b = n // 2
    kernel = np.zeros((n, n), dtype=float)
    for x in range(n):
        for y in range(n):
            exponent = -((x - a)**2 + (y - b)**2) / (2 * sigma**2)
            kernel[x, y] = (1 / (2 * math.pi * sigma**2)) * math.exp(exponent)
    kernel /= np.sum(kernel)
    return kernel

def gaussian_blur(image, kernel):
    h, w = image.shape
    n = kernel.shape[0]
    offset = n // 2
    result = np.zeros_like(image, dtype=float)
    for i in range(offset, h - offset):
        for j in range(offset, w - offset):
            region = image[i - offset:i + offset + 1, j - offset:j + offset + 1]
            result[i, j] = np.sum(region * kernel)
    return np.uint8(result)

def task1(image_path: str):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (480, 360))

    kernel = gaussian_kernel(5, sigma=5)
    blurred = gaussian_blur(gray_resized, kernel)

    return blurred

def task2(blurred_image):
    Gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=float)
    Gy_kernel = np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]], dtype=float)

    h, w = blurred_image.shape
    Gx = np.zeros_like(blurred_image, dtype=float)
    Gy = np.zeros_like(blurred_image, dtype=float)
    offset = 1

    for i in range(offset, h - offset):
        for j in range(offset, w - offset):
            region = blurred_image[i - offset:i + offset + 1, j - offset:j + offset + 1]
            Gx[i, j] = np.sum(region * Gx_kernel)
            Gy[i, j] = np.sum(region * Gy_kernel)

    magnitude = np.sqrt(Gx**2 + Gy**2)
    angle = np.arctan2(Gy, Gx) * (180.0 / np.pi)

    return Gx, Gy, magnitude, angle

def task3(magnitude, angle):
    h, w = magnitude.shape
    suppressed = np.zeros((h, w), dtype=np.float32)

    angle = angle % 180  # Угол в пределах 0-180

    for i in range(1, h-1):
        for j in range(1, w-1):
            q = 255
            r = 255

            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if magnitude[i,j] >= q and magnitude[i,j] >= r:
                suppressed[i,j] = magnitude[i,j]
            else:
                suppressed[i,j] = 0

    suppressed_display = cv2.normalize(suppressed, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("Non-maximum suppression", np.uint8(suppressed_display))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return suppressed


blurred = task1("C:/imageLab/akula.jpg")
Gx, Gy, magnitude, angle = task2(blurred)
suppressed = task3(magnitude, angle)
