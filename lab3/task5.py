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

def gaussian_blur_manual(image, kernel):
    h, w = image.shape
    n = kernel.shape[0]
    offset = n // 2
    result = np.zeros_like(image, dtype=float)
    for i in range(offset, h - offset):
        for j in range(offset, w - offset):
            region = image[i - offset:i + offset + 1, j - offset:j + offset + 1]
            result[i, j] = np.sum(region * kernel)
    return np.uint8(result)


img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
resized_img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA)

kernel_size = 7
sigma = 15


kernel = gaussian_kernel(kernel_size, sigma)
blur_manual = gaussian_blur_manual(resized_img, kernel)


blur_opencv = cv2.GaussianBlur(resized_img, (kernel_size, kernel_size), sigmaX=sigma)

cv2.imshow("Original", resized_img)
cv2.imshow("Manual Gaussian Blur", blur_manual)
cv2.imshow("OpenCV Gaussian Blur", blur_opencv)

cv2.waitKey(0)
cv2.destroyAllWindows()
