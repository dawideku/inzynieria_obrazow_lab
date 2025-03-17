import cv2
import numpy as np
import matplotlib.pyplot as plt

loaded_image = cv2.imread('test_image.png')
loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)

def zadanie_1(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    # Zastosowanie filtru
    filtered_image = cv2.filter2D(image, -1, kernel)

    # Wyświetlanie obu obrazów
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Oryginalny obraz')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Po filtracji')
    plt.imshow(filtered_image)
    plt.axis('off')

    plt.show()

def zadanie_2(image):
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
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Oryginalny obraz')
    plt.imshow(image_float)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Po transformacji')
    plt.imshow(transformed_image)
    plt.axis('off')

    plt.show()

def zadanie_3(image):
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

def zadanie_4_5(image):
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


while True:
    print("Wpisz numer zadania:")
    print("1 - Zadanie 1")
    print("2 - Zadanie 2")
    print("3 - Zadanie 3")
    print("4 - Zadanie 4 oraz zadanie 5")
    option = input()
    if option == "1":
        zadanie_1(loaded_image)
    elif option == "2":
        zadanie_2(loaded_image)
    elif option == "3":
        zadanie_3(loaded_image)
    elif option == "4":
        zadanie_4_5(loaded_image)
    else:
        print("Podano złą wartość")