from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math

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
    array = [array[i:i + 8] for i in range(0, len(array), 8)]
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


def hide_message(image, message, nbits=1, spos=0):
    """Hide a message in an image (LSB) starting from spos."""
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()

    if spos < 0 or spos >= len(image):
        raise ValueError("Start position out of bounds.")

    if len(message) > (len(image) - spos) * nbits:
        raise ValueError("Message is too long for the given start position.")

    chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]

    for i, chunk in enumerate(chunks):
        index = spos + i
        byte = "{:08b}".format(image[index])
        new_byte = byte[:-nbits] + chunk
        image[index] = int(new_byte, 2)

    return image.reshape(shape)


def reveal_message(image, nbits=1, length=0, spos=0):
    """Reveal a hidden message in an image (LSB) starting from spos."""
    nbits = clamp(nbits, 1, 8)
    image = np.copy(image).flatten()

    if spos < 0 or spos >= len(image):
        raise ValueError("Start position out of bounds.")

    length_in_pixels = math.ceil(length / nbits)
    end_pos = spos + length_in_pixels
    if end_pos > len(image):
        raise ValueError("Message exceeds image bounds.")

    message = ""
    for i in range(spos, spos + length_in_pixels):
        byte = "{:08b}".format(image[i])
        message += byte[-nbits:]

    mod = length % -nbits
    if mod != 0:
        message = message[:mod]

    return message


image_path = "images/test_image.png"
original_img = load_image(image_path)

# Przykładowa wiadomość
message = "To jest ukryta wiadomość z przesunięciem!"
binary_message = encode_as_binary_array(message)

# Ukrycie wiadomości od pozycji 1000 z 2 najmłodszymi bitami
encoded_img = hide_message(original_img, binary_message, nbits=2, spos=1000)

# Zapis obrazu
save_image("images/hidden_from_1000.png", encoded_img)

# Odczyt wiadomości
extracted_binary = reveal_message(encoded_img, nbits=2, length=len(binary_message), spos=1000)
extracted_message = decode_from_binary_array(extracted_binary)

print("Odczytana wiadomość:", extracted_message)
