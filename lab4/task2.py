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
    # Операторы Собеля
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

    # Нормализация для визуализации
    magnitude_display = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    angle_display = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow("Gradient magnitude", np.uint8(magnitude_display))
    cv2.imshow("Gradient angle", np.uint8(angle_display))

    print("Matrix of gradient magnitudes:\n", magnitude)
    print("\nMatrix of gradient angles (degrees):\n", angle)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


blurred = task1("C:/imageLab/akula.jpg")
task2(blurred)
