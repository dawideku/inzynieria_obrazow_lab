import numpy as np
import matplotlib.pyplot as plt
import binascii
import cv2 as cv
import math
import lorem

plt.rcParams["figure.figsize"] = (18, 10)

def encode_as_binary_array(msg):
    msg = msg.encode("utf-8").hex()
    msg = ["{:08b}".format(int(msg[i:i+2], 16)) for i in range(0, len(msg), 2)]
    return "".join(msg)

def decode_from_binary_array(array):
    array = [array[i:i+8] for i in range(0, len(array), 8)]
    if len(array[-1]) != 8:
        array[-1] += "0" * (8 - len(array[-1]))
    array = ["{:02x}".format(int(el, 2)) for el in array]
    return binascii.unhexlify("".join(array)).decode("utf-8", errors="replace")

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def load_image(path, pad=False):
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    if pad:
        y_pad = 8 - (image.shape[0] % 8)
        x_pad = 8 - (image.shape[1] % 8)
        image = np.pad(image, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant')
    return image

def save_image(path, image):
    plt.imsave(path, image)

def hide_message(image, message, nbits=1, spread_factor=2.0):
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()

    if len(message) > len(image) * nbits:
        raise ValueError("Message is too long :(")

    chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]

    # Ustal liczbę pikseli, w których będzie rozrzucona wiadomość
    used_pixels = min(int(len(chunks) * spread_factor), len(image))
    positions = np.linspace(0, len(image) - 1, used_pixels, dtype=int)[:len(chunks)]

    for i, chunk in zip(positions, chunks):
        byte = "{:08b}".format(image[i])
        image[i] = int(byte[:-nbits] + chunk, 2)

    return image.reshape(shape)


def reveal_message(image, nbits=1, length=0):
    nbits = clamp(nbits, 1, 8)
    image = np.copy(image).flatten()
    length_in_pixels = math.ceil(length / nbits)
    if length_in_pixels > len(image) or length_in_pixels <= 0:
        length_in_pixels = len(image)
    message = ""
    for i in range(length_in_pixels):
        message += "{:08b}".format(image[i])[-nbits:]
    mod = length % -nbits
    if mod != 0:
        message = message[:mod]
    return message

# ---------- GŁÓWNA LOGIKA -----------------

image_path = "images/test_image.png"
original = load_image(image_path)
flat_pixels = original.size

mse_values = []

for n in range(1, 9):
    needed_bits = int(flat_pixels * 0.75 / n) * n  # dla 75% obrazka
    needed_bytes = math.ceil(needed_bits / 8)
    sample = lorem.text() * 6
    example_text = (sample * ((needed_bytes // len(sample)) + 1))[:needed_bytes]
    binary_message = encode_as_binary_array(example_text)

    encoded_img = hide_message(original, binary_message, nbits=n)
    save_path = f"images/with_message_nbits_{n}.png"
    save_image(save_path, encoded_img)

    mse = np.mean((original.astype(np.float32) - encoded_img.astype(np.float32)) ** 2)
    mse_values.append(mse)
    print(f"nbits={n}, MSE={mse:.2f}, saved to {save_path}")

# ---------- WYKRES -----------------

plt.figure()
plt.plot(range(1, 9), mse_values, marker='o')
plt.title("MSE vs nbits")
plt.xlabel("nbits")
plt.ylabel("MSE")
plt.grid(True)
plt.savefig("images/mse_plot.png")
plt.show()
