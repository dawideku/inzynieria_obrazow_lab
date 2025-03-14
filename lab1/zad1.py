import cv2
import numpy as np
import matplotlib.pyplot as plt

# Wczytaj obraz w kolorze
image = cv2.imread('piramidy.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konwersja do RGB

# Definiowanie maski filtru górnoprzepustowego
kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])

# Zastosowanie filtru dla każdej składowej koloru
filtered_image = cv2.filter2D(image, -1, kernel)

# Wyświetlenie oryginalnego i przefiltrowanego obrazu
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Oryginalny obraz')
plt.imshow(image)
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Po filtracji')
plt.imshow(filtered_image)
plt.axis('off')

plt.show()
