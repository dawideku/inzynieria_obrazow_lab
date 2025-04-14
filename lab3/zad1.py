from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math

plt.rcParams["figure.figsize"] = (18, 10)


def encode_as_binary_array(msg):
    msg = msg.encode("utf-8")
    msg = msg.hex()
    msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]
    msg = ["{:08b}".format(int(el, base=16)) for el in msg]
    return "".join(msg)


def decode_from_binary_array(array):
    array = [array[i:i + 8] for i in range(0, len(array), 8)]
    if len(array[-1]) != 8:
        array[-1] = array[-1] + "0" * (8 - len(array[-1]))
    array = ["{:02x}".format(int(el, 2)) for el in array]
    array = "".join(array)
    result = binascii.unhexlify(array)
    return result.decode("utf-8", errors="replace")


def load_image(path, pad=False):
    image = cv.imread(path)
    if image is None:
        raise FileNotFoundError(f"Nie znaleziono pliku obrazu: {path}")
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    if pad:
        y_pad = 8 - (image.shape[0] % 8)
        x_pad = 8 - (image.shape[1] % 8)
        image = np.pad(image, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant')
    return image


def save_image(path, image):
    plt.imsave(path, image)


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def hide_message(image, message, nbits=1):
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    if len(message) > len(image) * nbits:
        raise ValueError("Wiadomość jest za długa, aby ją ukryć w tym obrazie.")
    chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]
    for i, chunk in enumerate(chunks):
        byte = "{:08b}".format(image[i])
        new_byte = byte[:-nbits] + chunk
        image[i] = int(new_byte, 2)
    return image.reshape(shape)


def reveal_message(image, nbits=1, length=0):
    nbits = clamp(nbits, 1, 8)
    image = np.copy(image).flatten()
    length_in_pixels = math.ceil(length / nbits)
    if len(image) < length_in_pixels or length_in_pixels <= 0:
        length_in_pixels = len(image)
    message = ""
    for i in range(length_in_pixels):
        byte = "{:08b}".format(image[i])
        message += byte[-nbits:]
    mod = length % -nbits
    if mod != 0:
        message = message[:mod]
    return message



obrazek_path = "images/test_image.png"
wiadomosc = "To jest tajna wiadomość ukryta w obrazku"
nbits = 1

obraz = load_image(obrazek_path)
binarna_wiadomosc = encode_as_binary_array(wiadomosc)
zakodowany_obraz = hide_message(obraz, binarna_wiadomosc, nbits=nbits)
save_image("images/obraz_z_wiadomoscia.png", zakodowany_obraz)

zakodowany_png = load_image("images/obraz_z_wiadomoscia.png")
odszyfrowany_bin = reveal_message(zakodowany_png, nbits=nbits, length=len(binarna_wiadomosc))
odszyfrowana_wiadomosc = decode_from_binary_array(odszyfrowany_bin)

print("\nOdszyfrowana wiadomość:")
print(odszyfrowana_wiadomosc)
