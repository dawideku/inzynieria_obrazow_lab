import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import binascii


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


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def hide_message(image, message, nbits=1, spos=0):
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    if len(message) + spos > len(image) * nbits:
        raise ValueError("Message is too long")
    chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]
    for i, chunk in enumerate(chunks):
        idx = i + spos
        byte = "{:08b}".format(image[idx])
        new_byte = byte[:-nbits] + chunk
        image[idx] = int(new_byte, 2)
    return image.reshape(shape)


def reveal_message(image, nbits=1, spos=0):
    nbits = clamp(nbits, 1, 8)
    image = np.copy(image).flatten()
    message = ""
    for i in range(spos, len(image)):
        byte = "{:08b}".format(image[i])
        message += byte[-nbits:]
    return message


def hide_image(carrier_image, secret_image_path, nbits=1):
    with open(secret_image_path, "rb") as file:
        secret_img = file.read()
    secret_img = secret_img.hex()
    secret_img = ["{:08b}".format(int(secret_img[i:i + 2], 16)) for i in range(0, len(secret_img), 2)]
    binary_secret = "".join(secret_img)
    return hide_message(carrier_image, binary_secret, nbits), binary_secret


def recover_image(carrier_image, nbits=1, signature="ffd9", output_path="images/recovered.jpg"):
    binary_data = reveal_message(carrier_image, nbits)
    # Podziel na bajty
    bytes_data = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]
    hex_data = ["{:02x}".format(int(b, 2)) for b in bytes_data if len(b) == 8]
    hex_str = "".join(hex_data)

    # Szukaj stopki obrazu JPG
    end_pos = hex_str.find(signature)
    if end_pos == -1:
        raise ValueError("End of image signature not found")
    end_pos += len(signature)
    final_data = hex_str[:end_pos]

    # Zapisz jako obraz
    binary_output = binascii.unhexlify(final_data)
    with open(output_path, "wb") as f:
        f.write(binary_output)
    print(f"Recovered image saved to: {output_path}")


# Załaduj obraz bazowy
carrier = load_image("images/test_image.png")

# Ukryj obraz .jpg
secret_path = "images/cat.jpg"
image_with_hidden, secret_bits = hide_image(carrier, secret_path, nbits=1)
save_image("images/with_hidden_image.png", image_with_hidden)
print("Image with hidden content saved.")

# Odzyskaj ukryty obraz
recover_image(image_with_hidden, nbits=1, signature="ffd9", output_path="images/recovered_image.jpg")
