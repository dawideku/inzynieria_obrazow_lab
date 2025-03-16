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

# Downsampling (2x2 subsampling)
Cb_down = Cb[::2, ::2]
Cr_down = Cr[::2, ::2]

# Upsampling (rozciąganie wartości do oryginalnego rozmiaru)
Cb_up = cv2.resize(Cb_down, (Cb.shape[1], Cb.shape[0]), interpolation=cv2.INTER_NEAREST)
Cr_up = cv2.resize(Cr_down, (Cr.shape[1], Cr.shape[0]), interpolation=cv2.INTER_NEAREST)

# Odtworzenie obrazu YCbCr po transmisji
ycbcr_reconstructed = cv2.merge([Y, Cr_up, Cb_up])

# Konwersja odwrotna YCbCr -> RGB
inverse_matrix = np.linalg.inv(ycbcr_matrix)
rgb_reconstructed = np.dot(ycbcr_reconstructed - [0, 128, 128], inverse_matrix.T)
rgb_reconstructed = np.clip(rgb_reconstructed, 0, 255).astype(np.uint8)

# Obliczanie błedu średniokwadratowego
def mse(original, reconstructed):
    return np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)

mse_Y = mse(Y, ycbcr_reconstructed[:, :, 0])
mse_Cb = mse(Cb, Cb_up)
mse_Cr = mse(Cr, Cr_up)
mse_total = mse(image, rgb_reconstructed)

print(f"MSE Y: {mse_Y:.4f}")
print(f"MSE Cb: {mse_Cb:.4f}")
print(f"MSE Cr: {mse_Cr:.4f}")
print(f"MSE Total: {mse_total:.4f}")

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
plt.title('Składowa Cb (Downsampled)')
plt.imshow(Cb_down, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Składowa Cr (Downsampled)')
plt.imshow(Cr_down, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Obraz po transmisji')
plt.imshow(rgb_reconstructed)
plt.axis('off')

plt.tight_layout()
plt.show()
