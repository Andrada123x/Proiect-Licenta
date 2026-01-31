import cv2
import numpy as np
from collections import deque


# 1. Filtrare Gaussiană

def gaussian_filter(img):
    return cv2.GaussianBlur(img, (5, 5), 1.4)


# 2. Gradient Sobel

def compute_gradient(img):
    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.arctan2(Gy, Gx) * 180 / np.pi
    direction[direction < 0] += 180

    return magnitude, direction

# 3. Suprimarea non-maximelor

def non_maximum_suppression(mag, direction):
    H, W = mag.shape
    result = np.zeros((H, W), dtype=np.float64)

    for i in range(1, H-1):
        for j in range(1, W-1):
            angle = direction[i, j]

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q, r = mag[i, j+1], mag[i, j-1]
            elif (22.5 <= angle < 67.5):
                q, r = mag[i-1, j+1], mag[i+1, j-1]
            elif (67.5 <= angle < 112.5):
                q, r = mag[i-1, j], mag[i+1, j]
            else:
                q, r = mag[i-1, j-1], mag[i+1, j+1]

            if mag[i, j] >= q and mag[i, j] >= r:
                result[i, j] = mag[i, j]

    return result


# 4. Binarizare adaptivă
\
def adaptive_threshold(nms, ratio=0.05):
    nms_norm = np.zeros_like(nms, dtype=np.uint8)

    if nms.max() != 0:
        nms_norm = np.clip(nms / nms.max() * 255, 0, 255).astype(np.uint8)

    # eliminăm marginile
    nms_norm[0, :] = 0
    nms_norm[-1, :] = 0
    nms_norm[:, 0] = 0
    nms_norm[:, -1] = 0

    hist = np.zeros(256, dtype=int)
    for v in nms_norm.flatten():
        hist[v] += 1

    non_zero = np.count_nonzero(nms_norm)
    NrNonMuchie = int((1 - ratio) * non_zero)

    s = 0
    threshold = 0
    for i in range(1, 256):
        s += hist[i]
        if s > NrNonMuchie:
            threshold = i
            break

    # imagine binarizată adaptiv
    bin_adaptiv = np.zeros_like(nms_norm)
    bin_adaptiv[nms_norm >= threshold] = 255

    return threshold, nms_norm, bin_adaptiv

 # 5. Histereză

def hysteresis(img, high_thresh, k=0.4):
    low_thresh = int(k * high_thresh)

    H, W = img.shape
    result = np.zeros((H, W), dtype=np.uint8)

    STRONG = 255
    WEAK = 128

    queue = deque()

    for i in range(H):
        for j in range(W):
            if img[i, j] >= high_thresh:
                result[i, j] = STRONG
                queue.append((i, j))
            elif img[i, j] >= low_thresh:
                result[i, j] = WEAK

    neighbors = [(-1,-1), (-1,0), (-1,1),
                 ( 0,-1),         ( 0,1),
                 ( 1,-1), ( 1,0), ( 1,1)]

    while queue:
        x, y = queue.popleft()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                if result[nx, ny] == WEAK:
                    result[nx, ny] = STRONG
                    queue.append((nx, ny))

    result[result != STRONG] = 0
    return result


if __name__ == "__main__":
    img = cv2.imread( r"C:\Users\User1\Desktop\Flower.jpg", cv2.IMREAD_GRAYSCALE)

    blur = gaussian_filter(img)
    mag, direction = compute_gradient(blur)
    nms = non_maximum_suppression(mag, direction)

    PragAdaptiv, nms_norm, bin_a = adaptive_threshold(nms, ratio=0.05)
    edges_b = hysteresis(nms_norm, PragAdaptiv, k=0.4)

    # AFIȘARE
    cv2.imshow("Imagine originala", img)
    cv2.imshow("a) Binarizare adaptiva", bin_a)
    cv2.imshow("b) Histereza - muchii finale", edges_b)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
