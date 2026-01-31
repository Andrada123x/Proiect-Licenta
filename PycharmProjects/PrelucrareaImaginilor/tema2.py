import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


DEFAULT_IMAGE_PATH = r'C:\Users\User1\Desktop\flower.jpg'


img = cv2.imread(DEFAULT_IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Imaginea nu a fost găsită. Verifică calea!")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

base_dir = os.path.dirname(DEFAULT_IMAGE_PATH)
output_dir = os.path.join(base_dir, "output_tema_lab_3")


os.makedirs(output_dir, exist_ok=True)


cv2.imshow("Imaginea originala", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

original_save_path = os.path.join(output_dir, "imagine_originala.jpg")
cv2.imwrite(original_save_path, img)


def global_threshold_auto(image, epsilon=0.5):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    min_intensity = np.min(np.where(hist > 0))
    max_intensity = np.max(np.where(hist > 0))
    T = (min_intensity + max_intensity) / 2
    while True:
        G1 = image[image <= T]
        G2 = image[image > T]
        if len(G1) == 0 or len(G2) == 0:
            break
        m1 = np.mean(G1)
        m2 = np.mean(G2)
        new_T = (m1 + m2) / 2
        if abs(new_T - T) < epsilon:
            break
        T = new_T
    _, binary_img = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
    return binary_img, T

binary_img, threshold_value = global_threshold_auto(gray)

print(f"Pragul calculat automat: {threshold_value:.2f}")
cv2.imshow(f"Imagine binarizata (prag = {threshold_value:.2f})", binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


binary_save_path = os.path.join(output_dir, f"imagine_binarizata_{int(threshold_value)}.jpg")
cv2.imwrite(binary_save_path, binary_img)


equalized_img = cv2.equalizeHist(gray)

cv2.imshow("Imagine egalizata (histograma uniformizata)", equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


equalized_save_path = os.path.join(output_dir, "imagine_egalizata.jpg")
cv2.imwrite(equalized_save_path, equalized_img)


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Histograma originala")
plt.hist(gray.ravel(), bins=256, range=[0,256], color='gray')
plt.subplot(1,2,2)
plt.title("Histograma egalizata")
plt.hist(equalized_img.ravel(), bins=256, range=[0,256], color='gray')
plt.tight_layout()

# Salvare figura cu histograme
hist_save_path = os.path.join(output_dir, "histograme_comparate.png")
plt.savefig(hist_save_path)

plt.show()

