import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct


# Wczytaj obraz i konwersja do YCbCr
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


# ⬇️ DCT i kwantyzacja ⬇️
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Kwantyzacja JPEG (standardowa macierz luminancji)
def get_quantization_matrix(QF):
    Q50 = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ])
    if QF < 50:
        scale = 5000 / QF
    else:
        scale = 200 - 2 * QF
    Q = np.floor((Q50 * scale + 50) / 100).astype(np.uint8)
    Q[Q == 0] = 1
    return Q


# ⛓️ Główna pipeline JPEG
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


# 🔍 Testujemy dla różnych QF
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
