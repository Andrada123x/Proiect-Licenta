import numpy as np
import cv2

IMAGE_PATH = r'C:\Users\User1\Desktop\flower.jpg'

def center_transform(img):
    """Transformare de centrare: x_uv = (-1)^(u+v) * x_uv"""
    u, v = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    return img * ((-1) ** (u + v))


def apply_filter(image, filter_type, param):

    H, W = image.shape


    centered = center_transform(image.astype(float))

    fft = np.fft.fft2(centered)

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    distance_sq = (H / 2 - u) ** 2 + (W / 2 - v) ** 2

    if filter_type == 'ideal_low':
        mask = (distance_sq <= param ** 2).astype(float)

    elif filter_type == 'gauss_low':
        mask = np.exp(-distance_sq / (param ** 2))

    elif filter_type == 'ideal_high':
        mask = (distance_sq > param ** 2).astype(float)

    elif filter_type == 'gauss_high':
        mask = 1 - np.exp(-distance_sq / (param ** 2))

    filtered_fft = fft * mask

    ifft = np.fft.ifft2(filtered_fft)

    result = center_transform(np.real(ifft))

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result, mask


def main():

    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    R = 30
    A = 50

    ideal_low, mask_il = apply_filter(img, 'ideal_low', R)
    gauss_low, mask_gl = apply_filter(img, 'gauss_low', A)
    ideal_high, mask_ih = apply_filter(img, 'ideal_high', R)
    gauss_high, mask_gh = apply_filter(img, 'gauss_high', A)

    # Salvare rezultate
    cv2.imwrite('output_ideal_lowpass.png', ideal_low)
    cv2.imwrite('output_gaussian_lowpass.png', gauss_low)
    cv2.imwrite('output_ideal_highpass.png', ideal_high)
    cv2.imwrite('output_gaussian_highpass.png', gauss_high)


if __name__ == "__main__":
    main()
