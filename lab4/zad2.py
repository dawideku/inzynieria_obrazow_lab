import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def find_closest_palette_color(value, k):
    return round((k - 1) * value / 255) * 255 / (k - 1)


def floyd_steinberg_dithering_color(image, k=2):
    pixels = np.array(image, dtype=np.float32)
    height, width, channels = pixels.shape

    for y in range(height):
        for x in range(width):
            old_pixel = pixels[y, x].copy()
            new_pixel = np.zeros(3)

            for c in range(3):
                new_pixel[c] = find_closest_palette_color(old_pixel[c], k)

            pixels[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            for c in range(3):
                if x + 1 < width:
                    pixels[y, x + 1, c] += quant_error[c] * 7 / 16
                if y + 1 < height and x > 0:
                    pixels[y + 1, x - 1, c] += quant_error[c] * 3 / 16
                if y + 1 < height:
                    pixels[y + 1, x, c] += quant_error[c] * 5 / 16
                if y + 1 < height and x + 1 < width:
                    pixels[y + 1, x + 1, c] += quant_error[c] * 1 / 16

    pixels = np.clip(pixels, 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))


def reduce_colors(image, k=2):
    pixels = np.array(image, dtype=np.float32)
    height, width, channels = pixels.shape
    for y in range(height):
        for x in range(width):
            for c in range(3):
                pixels[y, x, c] = find_closest_palette_color(pixels[y, x, c], k)
    pixels = np.clip(pixels, 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))


def plot_histogram(image, title="Histogram"):
    pixels = np.array(image)
    colors = ('r', 'g', 'b')
    plt.figure(figsize=(10, 5))
    for i, color in enumerate(colors):
        hist, bins = np.histogram(pixels[:, :, i], bins=256, range=(0, 255))
        plt.plot(hist, color=color)
    plt.title(title)
    plt.xlabel("Wartość koloru")
    plt.ylabel("Liczba pikseli")
    plt.show()


img = Image.open("images/color_image.jfif").convert("RGB")

# Samo ograniczenie kolorów (bez ditheringu)
reduced_img = reduce_colors(img, k=2)
reduced_img.save("reduced_output.png")
reduced_img.show()

# Dithering kolorowy
dithered_img = floyd_steinberg_dithering_color(img, k=2)
dithered_img.save("dithered_output.png")
dithered_img.show()

# Histogramy
plot_histogram(reduced_img, title="Histogram po redukcji kolorów")
plot_histogram(dithered_img, title="Histogram po ditheringu")
