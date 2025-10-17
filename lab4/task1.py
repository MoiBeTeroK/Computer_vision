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
    cv2.imshow("Grayscale image", gray_resized)

    kernel = gaussian_kernel(5, sigma=5)
    blurred = gaussian_blur(gray_resized, kernel)
    cv2.imshow("Gaussian blur", blurred)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


task1("C:/imageLab/akula.jpg")
