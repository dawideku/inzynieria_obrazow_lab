"""Function definitions that are used in LSB steganography."""

from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math
from scipy.fftpack import dct, idct

plt.rcParams["figure.figsize"] = (18, 10)


def encode_as_binary_array(msg):
    """Encode a message as a binary string."""
    msg = msg.encode("utf-8")
    msg = msg.hex()
    msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]
    msg = ["{:08b}".format(int(el, base=16)) for el in msg]
    return "".join(msg)


def decode_from_binary_array(array):
    """Decode a binary string to utf8."""
    array = [array[i:i+8] for i in range(0, len(array), 8)]
    if len(array[-1]) != 8:
        array[-1] = array[-1] + "0" * (8 - len(array[-1]))
    array = ["{:02x}".format(int(el, 2)) for el in array]
    array = "".join(array)
    result = binascii.unhexlify(array)
    return result.decode("utf-8", errors="replace")


def load_image(path, pad=False):
    """Load an image.
    If pad is set then pad an image to multiple of 8 pixels.
    """
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    if pad:
        y_pad = 8 - (image.shape[0] % 8)
        x_pad = 8 - (image.shape[1] % 8)
        image = np.pad(
            image, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant'
        )
    return image


def save_image(path, image):
    """Save an image."""
    plt.imsave(path, image)


def clamp(n, minn, maxn):
    """Clamp the n value to be in range (minn, maxn)."""
    return max(min(maxn, n), minn)


def hide_message(image, message, nbits=1):
    """Hide a message in an image (LSB).
    nbits: number of least significant bits
    """
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    if len(message) > len(image) * nbits:
        raise ValueError("Message is too long :(")
    chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]
    for i, chunk in enumerate(chunks):
        byte = "{:08b}".format(image[i])
        new_byte = byte[:-nbits] + chunk
        image[i] = int(new_byte, 2)
    return image.reshape(shape)


def reveal_message(image, nbits=1, length=0):
    """Reveal the hidden message.
    nbits: number of least significant bits
    length: length of the message in bits.
    """
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    length_in_pixels = math.ceil(length / nbits)
    if len(image) < length_in_pixels or length_in_pixels <= 0:
        length_in_pixels = len(image)
    message = ""
    i = 0
    while i < length_in_pixels:
        byte = "{:08b}".format(image[i])
        message += byte[-nbits:]
        i += 1
    mod = length % -nbits
    if mod != 0:
        message = message[:mod]
    return message

original_image = load_image("images/test_image.png")  # Wczytanie obrazka

# Mnożenie stringów działa jak zwielokratnianie
message = "Czasie największej rozkoszy w życiu mego narodu - od otwarcia sklepów do ich zamknięcia!" * 1

n = 1  # liczba najmłodszych bitów używanych do ukrycia wiadomości

message = encode_as_binary_array(message)  # Zakodowanie wiadomości jako ciąg 0 i 1

image_with_message = hide_message(original_image, message, n)  # Ukrycie wiadomości w obrazku

# Zapisanie obrazka w formacie PNG i JPG
save_image("images/image_with_message.png", image_with_message)
save_image("images/image_with_message.jpg", image_with_message)

# Wczytanie zapisanych obrazków
image_with_message_png = load_image("images/image_with_message.png")
image_with_message_jpg = load_image("images/image_with_message.jpg")

# Odczytanie ukrytej wiadomości z obu wersji
secret_message_png = decode_from_binary_array(
    reveal_message(image_with_message_png, nbits=n, length=len(message))
)
secret_message_jpg = decode_from_binary_array(
    reveal_message(image_with_message_jpg, nbits=n, length=len(message))
)

print(secret_message_png)
print(secret_message_jpg)

# Wyświetlenie obrazków
f, ar = plt.subplots(2, 2)
ar[0, 0].imshow(original_image)
ar[0, 0].set_title("Original image")

ar[0, 1].imshow(image_with_message)
ar[0, 1].set_title("Image with message")

ar[1, 0].imshow(image_with_message_png)
ar[1, 0].set_title("PNG image")

ar[1, 1].imshow(image_with_message_jpg)
ar[1, 1].set_title("JPG image")

plt.tight_layout()
plt.show()

QY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 48, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float64)
QY = np.ceil(QY / 2)


def dct2(array):
    """Discrete cosine transform."""
    return dct(dct(array, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(array):
    """Inverse discrete cosine transform."""
    return idct(idct(array, axis=0, norm='ortho'), axis=1, norm='ortho')


def split_channel_to_blocks(channel):
    """Splits channel into blocks 8x8"""
    blocks = []
    for i in range(0, channel.shape[0], 8):
        for j in range(0, channel.shape[1], 8):
            blocks.append(channel[i:i + 8, j:j + 8])
    return blocks


def merge_blocks_to_channel(blocks, width):
    """Merge 8x8 blocks into a single channel."""
    step = int(width / 8)
    rows = []
    for i in range(0, len(blocks), step):
        rows.append(np.concatenate(blocks[i:i + step], axis=1))
    channel = np.concatenate(rows, axis=0)
    return channel


def hide_message(blocks, message):
    """Hide a message in blocks."""
    blocks = [b.astype(np.int64) for b in blocks]  # <- zmienione z int32 na int64
    i = 0
    for nb in range(len(blocks)):
        for x, y in [(x, y) for x in range(8) for y in range(8)]:
            value = blocks[nb][x, y]
            if i >= len(message):
                break
            if value == 0 or value == 1:
                continue
            m = message[i]
            i += 1
            v = np.binary_repr(value, width=32)
            new_value = int(v[:-1] + m, 2)  # <- użycie new_value
            blocks[nb][x, y] = new_value   # <- przypisanie new_value
        if i >= len(message):
            break
    if i < len(message):
        print("Could not encode whole message")
    return blocks



def reveal_message(blocks, length=0):
    """Reveal message from blocks."""
    blocks = [b.astype(np.int32) for b in blocks]
    message = ""
    i = 0
    for block in blocks:
        for x, y in [(x, y) for x in range(8) for y in range(8)]:
            value = block[x, y]
            if value == 0 or value == 1:
                continue
            message += np.binary_repr(value, width=32)[-1]
            i += 1
            if i >= length:
                return message
    return message


def y_to_dct_blocks(Y):
    """Convert Y to quantized dct blocks."""
    Y = Y.astype(np.float32)
    blocks = split_channel_to_blocks(Y)
    blocks = [dct2(block) for block in blocks]
    blocks = [np.round(block / QY) for block in blocks]
    return blocks


def dct_blocks_to_y(blocks, image_width):
    """Convert quantized dct blocks to Y."""
    blocks = [block * QY for block in blocks]
    blocks = [idct2(block) for block in blocks]
    Y = merge_blocks_to_channel(blocks, image_width).round()
    return Y


# Wczytanie obrazu i przygotowanie danych
original_image = load_image("images/test_image.png", True)
message = "Ala ma kota"

# Konwersja do YCbCr
image = cv.cvtColor(original_image, cv.COLOR_RGB2YCrCb)

# Podział na kanały
Y = image[:, :, 0]
Cr = image[:, :, 1]
Cb = image[:, :, 2]

# Konwersja Y na bloki DCT
blocks = y_to_dct_blocks(Y)

# Ukrycie wiadomości
encoded_msg = encode_as_binary_array(message)
blocks = hide_message(blocks, encoded_msg)

# Odczyt wiadomości z bloków
message_from_dct = reveal_message(blocks, len(encoded_msg))
print("Message from dct:", decode_from_binary_array(message_from_dct))

# Odzyskanie kanału Y
Y = dct_blocks_to_y(blocks, image.shape[1])

# Połączenie kanałów
image = np.stack([Y, Cr, Cb], axis=2)

# Konwersja do RGB i zapis
image = np.clip(image, 0, 255).astype(np.uint8)
image_with_message = cv.cvtColor(image, cv.COLOR_YCrCb2RGB)
save_image("images/image_with_message2.jpg", image_with_message)

# Wczytanie zapisanego obrazu
loaded_image_with_message = load_image("images/image_with_message2.jpg", True)

# Odczytanie wiadomości z JPEG
loaded_image_with_message = cv.cvtColor(loaded_image_with_message, cv.COLOR_RGB2YCrCb)
Y = loaded_image_with_message[:, :, 0]
blocks = y_to_dct_blocks(Y)
message_from_dct = reveal_message(blocks, len(encoded_msg))
print("Message from JPEG file:", decode_from_binary_array(message_from_dct))

# Wyświetlenie obrazków
f, ar = plt.subplots(3, dpi=150)
f.tight_layout()
ar[0].imshow(original_image)
ar[0].set_title("Original image")

ar[1].imshow(image_with_message)
ar[1].set_title("Image with message")

ar[2].imshow(loaded_image_with_message)
ar[2].set_title("Image with message loaded from JPEG")

plt.show()
