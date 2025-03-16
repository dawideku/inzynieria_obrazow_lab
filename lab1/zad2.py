import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('test_image.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Normalizacja do zakresu
image_float = image / 255.0

transformation_matrix = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.689, 0.164],
    [0.272, 0.534, 0.131]
])

# Transformacja każdego piksela
transformed_image = np.dot(image_float, transformation_matrix.T)

# Ograniczenie wartości do maksymalnie 1.0
transformed_image = np.clip(transformed_image, 0, 1)

# Wyświetlenie oryginalnego i przekształconego obrazu
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Oryginalny obraz')
plt.imshow(image_float)
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Po transformacji')
plt.imshow(transformed_image)
plt.axis('off')

plt.show()
