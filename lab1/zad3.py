import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('test_image.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Definicja macierzy transformacji RGB -> YCbCr
ycbcr_matrix = np.array([
    [0.229, 0.587, 0.114],
    [0.500, -0.418, -0.082],
    [-0.168, -0.331, 0.500]
])

# Konwersja obrazu RGB na YCbCr
ycbcr_image = np.dot(image, ycbcr_matrix.T) + [0, 128, 128]
Y, Cr, Cb = cv2.split(ycbcr_image.astype(np.uint8))

# Definicja macierzy odwrotnej (YCbCr -> RGB)
inverse_matrix = np.linalg.inv(ycbcr_matrix)

# Konwersja odwrotna YCbCr -> RGB
rgb_reconstructed = np.dot(ycbcr_image - [0, 128, 128], inverse_matrix.T)
rgb_reconstructed = np.clip(rgb_reconstructed, 0, 255).astype(np.uint8)

# Wyświetlenie obrazów
plt.figure(figsize=(10, 8))

plt.subplot(2, 3, 1)
plt.title('Oryginalny RGB')
plt.imshow(image)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Składowa Y')
plt.imshow(Y, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Składowa Cb')
plt.imshow(Cb, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Składowa Cr')
plt.imshow(Cr, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Rekonstrukcja RGB')
plt.imshow(rgb_reconstructed)
plt.axis('off')

plt.tight_layout()
plt.show()
