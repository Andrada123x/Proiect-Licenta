import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

IMG_PATH = r'C:\Users\User1\Desktop\flower.jpg'
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img_color = cv2.imread(IMG_PATH)
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
fdp = hist / np.sum(hist)

WH = 5
TH = 0.0003
maxime = []

for k in range(WH, 256 - WH):
    v = np.mean(fdp[k - WH:k + WH + 1])
    if fdp[k] > v + TH and fdp[k] >= np.max(fdp[k - WH:k + WH + 1]):
        maxime.append(k)

maxime = [0] + maxime + [255]
maxime = sorted(list(set(maxime)))  # elimină duplicate

print("Valorile pragurilor (maxime):", maxime)

praguri = [(maxime[i] + maxime[i + 1]) / 2 for i in range(len(maxime) - 1)]
praguri = [int(p) for p in praguri]

quantized = np.zeros_like(gray)
for val in range(256):

    idx = np.argmin([abs(val - m) for m in maxime])
    quantized[gray == val] = maxime[idx]

cv2.imwrite(os.path.join(OUTPUT_DIR, "floare_cuantizata.jpg"), quantized)

floyd = gray.astype(float).copy()
for y in range(floyd.shape[0] - 1):
    for x in range(1, floyd.shape[1] - 1):
        old_pixel = floyd[y, x]
        new_pixel = min(maxime, key=lambda m: abs(m - old_pixel))
        floyd[y, x] = new_pixel
        eroare = old_pixel - new_pixel

        floyd[y, x + 1] += 7 * eroare / 16
        floyd[y + 1, x - 1] += 3 * eroare / 16
        floyd[y + 1, x] += 5 * eroare / 16
        floyd[y + 1, x + 1] += eroare / 16

floyd = np.clip(floyd, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(OUTPUT_DIR, "floare_floyd.jpg"), floyd)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(quantized, cmap="gray")
plt.title("Cuantizare pe baza maximelor")

plt.subplot(1, 2, 2)
plt.imshow(floyd, cmap="gray")
plt.title("Floyd–Steinberg ")

plt.show()
