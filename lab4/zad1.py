import numpy as np
from PIL import Image

def find_closest_palette_color(value, levels = 6):
    return round((levels - 1) * value / 255) * (255 // (levels - 1))

def floyd_steinberg_dithering(image):
    pixels = np.array(image, dtype=np.float32)
    height, width = pixels.shape

    for y in range(height):
        for x in range(width):
            old_pixel = pixels[y, x]
            new_pixel = find_closest_palette_color(old_pixel)
            pixels[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            if x + 1 < width:
                pixels[y, x + 1] += quant_error * 7 / 16
            if y + 1 < height and x > 0:
                pixels[y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < height:
                pixels[y + 1, x] += quant_error * 5 / 16
            if y + 1 < height and x + 1 < width:
                pixels[y + 1, x + 1] += quant_error * 1 / 16
    pixels = np.clip(pixels, 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))

img = Image.open("images/gray_photo.jfif").convert("L")
dithered_img = floyd_steinberg_dithering(img)
dithered_img.save("dithered_output.png")
dithered_img.show()
