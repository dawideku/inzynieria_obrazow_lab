import numpy as np
import struct
import zlib


def generate_rainbow_bar():
    width, height = 120, 8
    image = np.zeros((height, width, 3), dtype=np.uint8)

    colors = [
        (0, 0, 0),       # czarny
        (255, 0, 0),     # niebieski (w BGR)
        (255, 255, 0),   # cyjan
        (0, 255, 0),     # zielony
        (0, 255, 255),   # żółty
        (0, 0, 255),     # czerwony
        (255, 0, 255),   # fioletowy
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

image = generate_rainbow_bar()
height, width, _ = image.shape

png_data = b''
for row in image:
    png_data += b'\x00' + row.tobytes()

compressed_data = zlib.compress(png_data)

def create_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack('!I', len(data))
    crc = struct.pack('!I', zlib.crc32(chunk_type + data))
    return length + chunk_type + data + crc

png_signature = b'\x89PNG\r\n\x1a\n'

ihdr_data = struct.pack('!IIBBBBB',
                        width,        # Szerokość
                        height,       # Wysokość
                        8,            # Głębia (8 bitów)
                        2,            # Kolor (2 = Truecolor)
                        0,            # Compression method
                        0,            # Filter method
                        0)            # Interlace method
ihdr_chunk = create_chunk(b'IHDR', ihdr_data)

idat_chunk = create_chunk(b'IDAT', compressed_data)

iend_chunk = create_chunk(b'IEND', b'')

with open('rainbow_output.png', 'wb') as f:
    f.write(png_signature)
    f.write(ihdr_chunk)
    f.write(idat_chunk)
    f.write(iend_chunk)

print("Zapisano plik PNG: rainbow_output.png")
