import numpy as np
import cv2
import matplotlib.pyplot as plt


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
plt.title("Wczytany pasek")
plt.show()
