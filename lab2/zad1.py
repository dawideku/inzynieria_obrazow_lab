import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def save_ppm_p3(filename, image):
    height, width, _ = image.shape
    with open(filename, 'w') as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for row in image:
            for pixel in row:
                f.write(f"{pixel[2]} {pixel[1]} {pixel[0]} ")
            f.write("\n")

def save_ppm_p6(filename, image):
    height, width, _ = image.shape
    with open(filename, 'wb') as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        f.write(image.tobytes())

def load_ppm_p3(filename):
    with open(filename, 'r') as f:
        assert f.readline().strip() == 'P3'
        width, height = map(int, f.readline().split())
        maxval = int(f.readline().strip())
        data = np.loadtxt(f, dtype=np.uint8).reshape((height, width, 3))
        return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

def load_ppm_p6(filename):
    with open(filename, 'rb') as f:
        assert f.readline().strip() == b'P6'
        width, height = map(int, f.readline().split())
        maxval = int(f.readline().strip())
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape((height, width, 3))
        return data

def compare_file_sizes(file1, file2):
    size1 = os.path.getsize(file1)
    size2 = os.path.getsize(file2)
    print(f"Rozmiar {file1}: {size1} bajtów")
    print(f"Rozmiar {file2}: {size2} bajtów")
    print(f"Różnica: {size1 - size2} bajtów")

image = np.zeros((100, 100, 3), dtype=np.uint8)
image[::2, ::2] = [255, 0, 0]
save_ppm_p3('test_p3.ppm', image)
save_ppm_p6('test_p6.ppm', image)
compare_file_sizes('test_p3.ppm', 'test_p6.ppm')

photo = cv2.imread('test_image.png')
save_ppm_p3('photo_p3.ppm', photo)
save_ppm_p6('photo_p6.ppm', photo)
compare_file_sizes('photo_p3.ppm', 'photo_p6.ppm')

loaded_p3 = load_ppm_p3('test_p3.ppm')
loaded_p6 = load_ppm_p6('test_p6.ppm')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(loaded_p3, cv2.COLOR_BGR2RGB))
axes[0].set_title("Loaded P3")
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(loaded_p6, cv2.COLOR_BGR2RGB))
axes[1].set_title("Loaded P6")
axes[1].axis("off")
plt.show()
