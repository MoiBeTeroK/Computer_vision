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
            val = np.sum(region * kernel)
            result[i, j] = val
    
    return np.uint8(result)


img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
resized_img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA)

for size in [3, 5, 7]:
    kernel = gaussian_kernel(size, sigma=20)
    print(f"\nГауссовское ядро {size}x{size}:")
    print(np.round(kernel, 10))
    print("Сумма элементов =", np.sum(kernel))
    
    blurred = gaussian_blur(resized_img, kernel)
    cv2.imshow(f"Gaussian Filter {size}x{size}", blurred)

cv2.imshow("Original", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
