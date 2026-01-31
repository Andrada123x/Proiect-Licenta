import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time


# ==================== FILTRARE GAUSSIAN ====================

def create_gaussian_kernel(size, sigma):
    """Creare nucleu Gaussian 2D"""
    center = size // 2
    kernel = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma * 2)) * np.exp(-(x * 2 + y * 2) / (2 * sigma * 2))

    # Normalizare
    kernel = kernel / np.sum(kernel)
    return kernel


def create_gaussian_1d(size, sigma):
    """Creare nucleu Gaussian 1D (pentru separabilitate)"""
    center = size // 2
    kernel = np.zeros(size)

    for i in range(size):
        x = i - center
        kernel[i] = (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-(x * 2) / (2 * sigma * 2))

    # Normalizare
    kernel = kernel / np.sum(kernel)
    return kernel


def convolve_2d(img, kernel):
    """Convoluție 2D (nucleu Gaussian bidimensional)"""
    h, w = img.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    # Padding
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
    result = np.zeros_like(img, dtype=np.float32)

    # Convoluție
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            result[i, j] = np.sum(region * kernel)

    return result.astype(np.uint8)


def convolve_separable(img, kernel_1d):
    """Convoluție separabilă (2 treceri cu nucleu 1D)"""
    h, w = img.shape
    k_size = len(kernel_1d)
    pad = k_size // 2

    # Prima trecere: convoluție pe linii (orizontal)
    padded = np.pad(img, ((0, 0), (pad, pad)), mode='edge')
    temp = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            temp[i, j] = np.sum(padded[i, j:j + k_size] * kernel_1d)

    # A doua trecere: convoluție pe coloane (vertical)
    padded = np.pad(temp, ((pad, pad), (0, 0)), mode='edge')
    result = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            result[i, j] = np.sum(padded[i:i + k_size, j] * kernel_1d)

    return result.astype(np.uint8)


def add_gaussian_noise(img, mean=0, sigma=25):
    """Adaugă zgomot Gaussian la imagine"""
    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


# ==================== OPERAȚII MORFOLOGICE ====================

def erosion(img, se):
    h, w = img.shape
    sh, sw = se.shape
    ph, pw = sh // 2, sw // 2

    padded = np.pad(img, ((ph, ph), (pw, pw)), constant_values=0)
    result = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + sh, j:j + sw]
            if np.all(region[se == 1] == 1):
                result[i, j] = 1

    return result * 255


def dilation(img, se):
    h, w = img.shape
    sh, sw = se.shape
    ph, pw = sh // 2, sw // 2

    binary = (img > 127).astype(np.uint8)
    result = np.zeros_like(binary)

    for i in range(h):
        for j in range(w):
            if binary[i, j] == 1:
                for si in range(sh):
                    for sj in range(sw):
                        if se[si, sj] == 1:
                            ni, nj = i + si - ph, j + sj - pw
                            if 0 <= ni < h and 0 <= nj < w:
                                result[ni, nj] = 1

    return result * 255


def extract_contour(img):
    se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    eroded = erosion(img, se)

    contour = img.astype(np.int16) - eroded.astype(np.int16)
    contour = np.clip(contour, 0, 255).astype(np.uint8)

    return contour


def region_filling(img, seed_point):
    se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    boundary = (img > 127).astype(np.uint8)
    complement = 1 - boundary

    X = np.zeros_like(boundary)
    X[seed_point[0], seed_point[1]] = 1

    for k in range(1000):
        X_prev = X.copy()

        X_dilated = dilation(X * 255, se)
        X_dilated = (X_dilated > 127).astype(np.uint8)

        X = X_dilated & complement

        if np.array_equal(X, X_prev):
            break

    return X * 255


def test_gaussian_2d():
    """Test filtrare cu nucleu Gaussian 2D"""
    # Citește imaginea
    img = Image.open(r'C:\Users\User1\Desktop\flower.jpg').convert('L')

    img = np.array(img)

    # Adaugă zgomot Gaussian
    noisy = add_gaussian_noise(img, sigma=25)

    # Parametri pentru filtrul Gaussian
    sigma = 1.0
    kernel_size = int(6 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Creare nucleu Gaussian 2D
    kernel_2d = create_gaussian_kernel(kernel_size, sigma)

    # Măsurare timp
    start = time.time()
    filtered = convolve_2d(noisy, kernel_2d)
    time_2d = time.time() - start

    # Vizualizare
    plt.figure(figsize=(15, 5))

    plt.subplot(141)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(142)
    plt.imshow(noisy, cmap='gray')
    plt.title('Cu zgomot Gaussian')
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(filtered, cmap='gray')
    plt.title(f'Filtrat 2D\n({time_2d:.3f}s)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('rezultat_gaussian_2d.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_gaussian_separable():
    img = Image.open(r'C:\Users\User1\Desktop\flower.jpg').convert('L')

    img = np.array(img)

    noisy = add_gaussian_noise(img, sigma=25)

    sigma = 1.0
    kernel_size = int(6 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel_1d = create_gaussian_1d(kernel_size, sigma)

    start = time.time()
    filtered = convolve_separable(noisy, kernel_1d)
    time_sep = time.time() - start

    # Vizualizare
    plt.figure(figsize=(15, 5))

    plt.subplot(141)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(142)
    plt.imshow(noisy, cmap='gray')
    plt.title('Cu zgomot Gaussian')
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(filtered, cmap='gray')
    plt.title(f'Filtrat Separabil\n({time_sep:.3f}s)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('rezultat_gaussian_separabil.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_gaussian_comparison():
    """Comparație între cele 2 metode"""
    img = Image.open('Flower.jpg').convert('L')
    img = np.array(img)

    noisy = add_gaussian_noise(img, sigma=25)

    sigma = 1.0
    kernel_size = int(6 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Metoda 1: 2D
    kernel_2d = create_gaussian_kernel(kernel_size, sigma)
    start = time.time()
    filtered_2d = convolve_2d(noisy, kernel_2d)
    time_2d = time.time() - start

    # Metoda 2: Separabilă
    kernel_1d = create_gaussian_1d(kernel_size, sigma)
    start = time.time()
    filtered_sep = convolve_separable(noisy, kernel_1d)
    time_sep = time.time() - start

    # Calculare diferență
    diff = np.abs(filtered_2d.astype(float) - filtered_sep.astype(float))

    plt.figure(figsize=(16, 4))

    plt.subplot(151)
    plt.imshow(noisy, cmap='gray')
    plt.title('Cu zgomot')
    plt.axis('off')

    plt.subplot(152)
    plt.imshow(filtered_2d, cmap='gray')
    plt.title(f'Filtrat 2D\nTimp: {time_2d:.3f}s')
    plt.axis('off')

    plt.subplot(153)
    plt.imshow(filtered_sep, cmap='gray')
    plt.title(f'Filtrat Separabil\nTimp: {time_sep:.3f}s')
    plt.axis('off')

    plt.subplot(154)
    plt.imshow(diff, cmap='hot')
    plt.title(f'Diferență absolută\nMax: {diff.max():.2f}')
    plt.colorbar(fraction=0.046)
    plt.axis('off')

    plt.subplot(155)
    speedup = time_2d / time_sep
    plt.bar(['2D', 'Separabil'], [time_2d, time_sep])
    plt.ylabel('Timp (secunde)')
    plt.title(f'Speedup: {speedup:.2f}x')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('rezultat_comparatie.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    test_gaussian_2d()
    test_gaussian_separable()
    test_gaussian_comparison()