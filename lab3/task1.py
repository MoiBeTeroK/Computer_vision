import numpy as np
import math

def gaussian_kernel(n, sigma):
    a = b = n // 2
    kernel = np.zeros((n, n), dtype=float)
    
    for x in range(n):
        for y in range(n):
            exponent = -((x - a)**2 + (y - b)**2) / (2 * sigma**2)
            kernel[x, y] = (1 / (2 * math.pi * sigma**2)) * math.exp(exponent)
    
    return kernel

for size in [3, 5, 7]:
    kernel = gaussian_kernel(size, sigma=1)
    print(f"\nГауссова матрица {size}x{size}:")
    print(np.round(kernel, 5))

    total = np.sum(kernel)
    print(f"Сумма всех элементов = {total:.5f}")
