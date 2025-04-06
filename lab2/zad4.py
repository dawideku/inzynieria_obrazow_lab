import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

img = Image.open("rainbow_output.png").convert("RGB")
img_np = np.array(img)

img_ycbcr = img.convert("YCbCr")
ycbcr_np = np.array(img_ycbcr)
Y, Cb, Cr = ycbcr_np[:, :, 0], ycbcr_np[:, :, 1], ycbcr_np[:, :, 2]


def subsample(channel, factor):
    if factor == 1:
        return channel
    else:
        return channel[::factor, ::factor]


def upsample(channel, original_shape):
    return np.array(Image.fromarray(channel).resize(original_shape[::-1], resample=Image.BILINEAR))


def split_into_blocks(channel):
    h, w = channel.shape
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    padded = np.pad(channel, ((0, h_pad), (0, w_pad)), mode='constant')
    blocks = padded.reshape(padded.shape[0] // 8, 8, padded.shape[1] // 8, 8).swapaxes(1, 2)
    return blocks


def zigzag(block):
    zigzag_order = sorted(
        ((x, y) for x in range(8) for y in range(8)),
        key=lambda p: (p[0] + p[1], -p[0] if (p[0] + p[1]) % 2 else p[0])
    )
    return np.array([block[x, y] for x, y in zigzag_order])


def encode_jpeg(y, cb, cr):
    img_encoded = Image.merge("YCbCr", (Image.fromarray(y), Image.fromarray(cb), Image.fromarray(cr))).convert("RGB")
    buffer = io.BytesIO()
    img_encoded.save(buffer, format="JPEG", quality=50)
    return buffer.getvalue(), img_encoded


def jpeg_pipeline(subsample_factor):
    cb_sub = subsample(Cb, subsample_factor)
    cr_sub = subsample(Cr, subsample_factor)

    cb_up = upsample(cb_sub, Y.shape)
    cr_up = upsample(cr_sub, Y.shape)

    y_blocks = split_into_blocks(Y)
    cb_blocks = split_into_blocks(cb_up)
    cr_blocks = split_into_blocks(cr_up)

    y_zigzag = np.array([zigzag(b) for row in y_blocks for b in row])
    cb_zigzag = np.array([zigzag(b) for row in cb_blocks for b in row])
    cr_zigzag = np.array([zigzag(b) for row in cr_blocks for b in row])

    encoded_bytes, reconstructed = encode_jpeg(Y, cb_up, cr_up)
    return len(encoded_bytes), reconstructed

sizes = {}
images = {}

for factor in [1, 2, 4]:
    size, recon = jpeg_pipeline(factor)
    sizes[factor] = size
    images[factor] = recon

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, factor in enumerate([1, 2, 4]):
    axs[i].imshow(images[factor])
    axs[i].set_title(f"Próbkowanie: co {factor}\nRozmiar: {sizes[factor]} B")
    axs[i].axis("off")
plt.tight_layout()
plt.show()
