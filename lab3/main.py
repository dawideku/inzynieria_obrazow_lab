import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import binascii
import cv2 as cv
import math
import lorem


def zadanie1():
    img = Image.open("rainbow_output.png").convert("RGB")
    img_ycbcr = img.convert("YCbCr")
    ycbcr_np = np.array(img_ycbcr)
    Y, Cb, Cr = ycbcr_np[:, :, 0], ycbcr_np[:, :, 1], ycbcr_np[:, :, 2]

    # Subsampowanie i upsampling
    def subsample(channel, factor):
        return channel if factor == 1 else channel[::factor, ::factor]

    def upsample(channel, original_shape):
        return np.array(Image.fromarray(channel).resize(original_shape[::-1], resample=Image.BILINEAR))

    # Dzielimy na bloki 8x8
    def split_into_blocks(channel):
        h, w = channel.shape
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8
        padded = np.pad(channel, ((0, h_pad), (0, w_pad)), mode='constant')
        blocks = padded.reshape(padded.shape[0] // 8, 8, padded.shape[1] // 8, 8).swapaxes(1, 2)
        return blocks

    # Zigzag nieużywany bez kodowania Huffmana – można zostawić
    def zigzag(block):
        zigzag_order = sorted(
            ((x, y) for x in range(8) for y in range(8)),
            key=lambda p: (p[0] + p[1], -p[0] if (p[0] + p[1]) % 2 else p[0])
        )
        return np.array([block[x, y] for x, y in zigzag_order])

    # DCT i kwantyzacja
    def dct2(block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def idct2(block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    # Kwantyzacja JPEG (standardowa macierz luminancji)
    def get_quantization_matrix(QF):
        Q50 = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])
        if QF < 50:
            scale = 5000 / QF
        else:
            scale = 200 - 2 * QF
        Q = np.floor((Q50 * scale + 50) / 100).astype(np.uint8)
        Q[Q == 0] = 1
        return Q

    # Główna pipeline JPEG
    def jpeg_pipeline(subsample_factor, QF):
        Q = get_quantization_matrix(QF)

        cb_sub = subsample(Cb, subsample_factor)
        cr_sub = subsample(Cr, subsample_factor)

        cb_up = upsample(cb_sub, Y.shape)
        cr_up = upsample(cr_sub, Y.shape)

        y_blocks = split_into_blocks(Y - 128)
        cb_blocks = split_into_blocks(cb_up - 128)
        cr_blocks = split_into_blocks(cr_up - 128)

        # DCT + Kwantyzacja
        def process_channel(blocks):
            processed = np.zeros_like(blocks)
            for i in range(blocks.shape[0]):
                for j in range(blocks.shape[1]):
                    dct_block = dct2(blocks[i, j])
                    quantized = np.round(dct_block / Q)
                    processed[i, j] = quantized
            return processed

        y_dct = process_channel(y_blocks)
        cb_dct = process_channel(cb_blocks)
        cr_dct = process_channel(cr_blocks)

        # Odtwarzanie (IDCT, dekwantyzacja)
        def reconstruct_channel(blocks):
            rec = np.zeros_like(blocks)
            for i in range(blocks.shape[0]):
                for j in range(blocks.shape[1]):
                    dequantized = blocks[i, j] * Q
                    idct_block = idct2(dequantized)
                    rec[i, j] = idct_block
            h, w = blocks.shape[0] * 8, blocks.shape[1] * 8
            return rec.swapaxes(1, 2).reshape(h, w) + 128

        y_rec = reconstruct_channel(y_dct)
        cb_rec = reconstruct_channel(cb_dct)
        cr_rec = reconstruct_channel(cr_dct)

        y_rec = np.clip(y_rec, 0, 255).astype(np.uint8)
        cb_rec = np.clip(cb_rec, 0, 255).astype(np.uint8)
        cr_rec = np.clip(cr_rec, 0, 255).astype(np.uint8)

        img_encoded = Image.merge("YCbCr", (
            Image.fromarray(y_rec),
            Image.fromarray(cb_rec),
            Image.fromarray(cr_rec)
        )).convert("RGB")

        # Zapis jako JPEG do bufora
        buffer = io.BytesIO()
        img_encoded.save(buffer, format="JPEG", quality=QF)
        return len(buffer.getvalue()), img_encoded

    # Testujemy dla różnych QF
    factors = [1, 2, 4]
    qf_values = [20, 50, 80]

    fig, axs = plt.subplots(len(factors), len(qf_values), figsize=(15, 10))
    for i, f in enumerate(factors):
        for j, q in enumerate(qf_values):
            size, img_out = jpeg_pipeline(f, q)
            axs[i, j].imshow(img_out)
            axs[i, j].axis("off")
            axs[i, j].set_title(f"Próbkowanie co {f}\nQF={q}\nRozmiar: {size} B")

    plt.tight_layout()
    plt.show()

def zadanie2():
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


def zadanie3():
    plt.rcParams["figure.figsize"] = (18, 10)

    def encode_as_binary_array(msg):
        msg = msg.encode("utf-8").hex()
        msg = ["{:08b}".format(int(msg[i:i + 2], 16)) for i in range(0, len(msg), 2)]
        return "".join(msg)

    def decode_from_binary_array(array):
        array = [array[i:i + 8] for i in range(0, len(array), 8)]
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

    def hide_message(image, message, nbits=1, spread_factor=1.0):
        nbits = clamp(nbits, 1, 8)
        shape = image.shape
        image = np.copy(image).flatten()

        if len(message) > len(image) * nbits:
            raise ValueError("Message is too long :(")

        chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]

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


    image_path = "images/test_image.png"
    original = load_image(image_path)
    flat_pixels = original.size

    mse_values = []

    for n in range(1, 9):
        needed_bytes = math.ceil(flat_pixels / 8)
        sample = lorem.text() * 6
        example_text = (sample * ((needed_bytes // len(sample)) + 1))[:needed_bytes]
        binary_message = encode_as_binary_array(example_text)

        encoded_img = hide_message(original, binary_message, nbits=n)
        save_path = f"images/with_message_nbits_{n}.png"
        save_image(save_path, encoded_img)

        mse = np.mean((original.astype(np.float32) - encoded_img.astype(np.float32)) ** 2)
        mse_values.append(mse)
        print(f"nbits={n}, MSE={mse:.2f}, saved to {save_path}")

    plt.figure()
    plt.plot(range(1, 9), mse_values, marker='o')
    plt.title("MSE vs nbits")
    plt.xlabel("nbits")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.savefig("images/mse_plot.png")
    plt.show()


def zadanie4():
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

def zadanie5():
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


def main():
    while True:
        print("\n=== MENU PROGRAMU ===")
        print("Informacja: na moim urządzeniu zadanie nr. 3 wykonywało się około 15-20 sekund")
        print("1. Zadanie 1 – Algorytm JPEG")
        print("2. Zadanie 2 – Ukrycie wiadomości")
        print("3. Zadanie 3 – Generowanie obrazków dla różnych nbits")
        print("4. Zadanie 4 – Odczytywanie wiadomości od zadanej pozycji")
        print("5. Zadanie 5 – Odzyskiwanie obrazka")
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
        elif choice == "5":
            zadanie5()
        elif choice == "0":
            print("Zamykam program.")
            break
        else:
            print("Niepoprawny wybór, spróbuj ponownie.")

if __name__ == "__main__":
    main()
