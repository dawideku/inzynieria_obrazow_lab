import numpy as np
import os
import struct
import zlib
import io
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys

# === ZADANIE 1 ===
def zadanie1():
    def save_ppm_p3(filename, image):
        height, width, _ = image.shape
        with open(filename, 'w') as f:
            f.write(f"P3\n{width} {height}\n255\n")
            for row in image:
                for pixel in row:
                    f.write(f"{pixel[2]} {pixel[1]} {pixel[0]} ")
                f.write("\n")

    def save_ppm_p6(filename, image):
        height, width, _ = image.shape
        with open(filename, 'wb') as f:
            f.write(f"P6\n{width} {height}\n255\n".encode())
            f.write(image.tobytes())

    def load_ppm_p3(filename):
        with open(filename, 'r') as f:
            assert f.readline().strip() == 'P3'
            width, height = map(int, f.readline().split())
            maxval = int(f.readline().strip())
            data = np.loadtxt(f, dtype=np.uint8).reshape((height, width, 3))
            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

    def load_ppm_p6(filename):
        with open(filename, 'rb') as f:
            assert f.readline().strip() == b'P6'
            width, height = map(int, f.readline().split())
            maxval = int(f.readline().strip())
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape((height, width, 3))
            return data

    def compare_file_sizes(file1, file2):
        size1 = os.path.getsize(file1)
        size2 = os.path.getsize(file2)
        print(f"Rozmiar {file1}: {size1} bajtów")
        print(f"Rozmiar {file2}: {size2} bajtów")
        print(f"Różnica: {size1 - size2} bajtów")

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[::2, ::2] = [255, 0, 0]
    save_ppm_p3('test_p3.ppm', image)
    save_ppm_p6('test_p6.ppm', image)
    compare_file_sizes('test_p3.ppm', 'test_p6.ppm')

    photo = cv2.imread('test_image.png')
    save_ppm_p3('photo_p3.ppm', photo)
    save_ppm_p6('photo_p6.ppm', photo)
    compare_file_sizes('photo_p3.ppm', 'photo_p6.ppm')

    loaded_p3 = load_ppm_p3('test_p3.ppm')
    loaded_p6 = load_ppm_p6('test_p6.ppm')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(loaded_p3, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Loaded P3")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(loaded_p6, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Loaded P6")
    axes[1].axis("off")
    plt.show()


# === ZADANIE 2 ===
def zadanie2():
    def save_ppm_p6(filename, image):
        height, width, _ = image.shape
        with open(filename, 'wb') as f:
            f.write(f"P6\n{width} {height}\n255\n".encode())
            f.write(image.tobytes())

    def load_ppm_p6(filename):
        with open(filename, 'rb') as f:
            assert f.readline().strip() == b'P6'
            width, height = map(int, f.readline().split())
            maxval = int(f.readline().strip())
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape((height, width, 3))
            return data

    def generate_rainbow_bar():
        width, height = 120, 8
        image = np.zeros((height, width, 3), dtype=np.uint8)

        colors = [
            (0, 0, 0),  # czarny
            (255, 0, 0),  # niebieski
            (255, 255, 0),  # cyjan
            (0, 255, 0),  # zielony
            (0, 255, 255),  # żółty
            (0, 0, 255),  # czerwony
            (255, 0, 255),  # fioletowy
            (255, 255, 255)  # biały
        ]

        gradient = np.zeros((width, 3), dtype=np.uint8)
        total_steps = width
        steps_per_color = total_steps // (len(colors) - 1)

        for i in range(len(colors) - 1):
            start_color = np.array(colors[i])
            end_color = np.array(colors[i + 1])
            for j in range(steps_per_color):
                t = j / (steps_per_color - 1)
                gradient[i * steps_per_color + j] = (1 - t) * start_color + t * end_color

        if total_steps % (len(colors) - 1) != 0:
            gradient[-(total_steps % (len(colors) - 1)):] = end_color

        for y in range(height):
            image[y] = gradient

        return image

    generated_image = generate_rainbow_bar()
    save_ppm_p6("rainbow.ppm", generated_image)
    print("Obraz zapisany jako rainbow.ppm w formacie P6.")
    loaded_image = load_ppm_p6("rainbow.ppm")
    plt.imshow(cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Wczytana tęcza")
    plt.show()


# === ZADANIE 3 ===
def zadanie3():
    def generate_rainbow_bar():
        width, height = 120, 8
        image = np.zeros((height, width, 3), dtype=np.uint8)

        colors = [
            (0, 0, 0),  # czarny
            (255, 0, 0),  # niebieski
            (255, 255, 0),  # cyjan
            (0, 255, 0),  # zielony
            (0, 255, 255),  # żółty
            (0, 0, 255),  # czerwony
            (255, 0, 255),  # fioletowy
            (255, 255, 255)  # biały
        ]

        gradient = np.zeros((width, 3), dtype=np.uint8)
        total_steps = width
        steps_per_color = total_steps // (len(colors) - 1)

        for i in range(len(colors) - 1):
            start_color = np.array(colors[i])
            end_color = np.array(colors[i + 1])
            for j in range(steps_per_color):
                t = j / (steps_per_color - 1)
                gradient[i * steps_per_color + j] = (1 - t) * start_color + t * end_color

        if total_steps % (len(colors) - 1) != 0:
            gradient[-(total_steps % (len(colors) - 1)):] = end_color

        for y in range(height):
            image[y] = gradient

        return image

    def create_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack('!I', len(data))
        crc = struct.pack('!I', zlib.crc32(chunk_type + data))
        return length + chunk_type + data + crc

    image = generate_rainbow_bar()
    height, width, _ = image.shape

    png_data = b''.join([b'\x00' + row.tobytes() for row in image])
    compressed_data = zlib.compress(png_data)

    png_signature = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('!IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr_chunk = create_chunk(b'IHDR', ihdr_data)
    idat_chunk = create_chunk(b'IDAT', compressed_data)
    iend_chunk = create_chunk(b'IEND', b'')

    with open('rainbow_output.png', 'wb') as f:
        f.write(png_signature)
        f.write(ihdr_chunk)
        f.write(idat_chunk)
        f.write(iend_chunk)

    print("Zapisano plik PNG: rainbow_output.png")


# === ZADANIE 4 ===
def zadanie4():
    img = Image.open("rainbow_output.png").convert("RGB")
    img_np = np.array(img)
    img_ycbcr = img.convert("YCbCr")
    ycbcr_np = np.array(img_ycbcr)
    Y, Cb, Cr = ycbcr_np[:, :, 0], ycbcr_np[:, :, 1], ycbcr_np[:, :, 2]

    def subsample(channel, factor):
        return channel if factor == 1 else channel[::factor, ::factor]

    def upsample(channel, original_shape):
        return np.array(Image.fromarray(channel).resize(original_shape[::-1], resample=Image.BILINEAR))

    def split_into_blocks(channel):
        h, w = channel.shape
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8
        padded = np.pad(channel, ((0, h_pad), (0, w_pad)), mode='constant')
        blocks = padded.reshape(padded.shape[0] // 8, 8, padded.shape[1] // 8, 8).swapaxes(1, 2)
        return blocks

    def zigzag(block):
        order = sorted(((x, y) for x in range(8) for y in range(8)),
                       key=lambda p: (p[0] + p[1], -p[0] if (p[0]+p[1]) % 2 else p[0]))
        return np.array([block[x, y] for x, y in order])

    def encode_jpeg(y, cb, cr):
        img_encoded = Image.merge("YCbCr", (Image.fromarray(y), Image.fromarray(cb), Image.fromarray(cr))).convert("RGB")
        buffer = io.BytesIO()
        img_encoded.save(buffer, format="JPEG", quality=50)
        return buffer.getvalue(), img_encoded

    def jpeg_pipeline(factor):
        cb_sub = subsample(Cb, factor)
        cr_sub = subsample(Cr, factor)
        cb_up = upsample(cb_sub, Y.shape)
        cr_up = upsample(cr_sub, Y.shape)

        y_blocks = split_into_blocks(Y)
        cb_blocks = split_into_blocks(cb_up)
        cr_blocks = split_into_blocks(cr_up)

        y_zigzag = np.array([zigzag(b) for row in y_blocks for b in row])
        cb_zigzag = np.array([zigzag(b) for row in cb_blocks for b in row])
        cr_zigzag = np.array([zigzag(b) for row in cr_blocks for b in row])

        encoded_bytes, reconstructed = encode_jpeg(Y, cb_up, cr_up)
        return len(encoded_bytes), reconstructed

    sizes = {}
    images = {}
    for factor in [1, 2, 4]:
        size, recon = jpeg_pipeline(factor)
        sizes[factor] = size
        images[factor] = recon

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, factor in enumerate([1, 2, 4]):
        axs[i].imshow(images[factor])
        axs[i].set_title(f"Próbkowanie: co {factor}\nRozmiar: {sizes[factor]} B")
        axs[i].axis("off")
    plt.tight_layout()
    plt.show()


# === MENU ===
def main():
    while True:
        print("\n=== MENU PROGRAMU ===")
        print("1. Zadanie 1 – Format PPM (P3 vs P6)")
        print("2. Zadanie 2 – Generowanie tęczy")
        print("3. Zadanie 3 – PNG z gradientem")
        print("4. Zadanie 4 – JPEG i próbkowanie chrominancji")
        print("0. Wyjście")

        choice = input("Wybierz numer zadania: ")
        if choice == "1":
            zadanie1()
        elif choice == "2":
            zadanie2()
        elif choice == "3":
            zadanie3()
        elif choice == "4":
            zadanie4()
        elif choice == "0":
            print("Zamykam program.")
            break
        else:
            print("Niepoprawny wybór, spróbuj ponownie.")

if __name__ == "__main__":
    main()
