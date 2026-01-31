import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

cale_imagine = r'C:\Users\User1\Desktop\flower.jpg'
imagine = cv2.imread(cale_imagine, cv2.IMREAD_GRAYSCALE)

nucleu_medie = (1/9) * np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.float32)

nucleu_gaussian = (1/16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)


nucleu_laplace = np.array([
    [0, -1,  0],
    [-1, 4, -1],
    [0, -1,  0]
], dtype=np.float32)

nucleu_highpass_4 = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
], dtype=np.float32)

filtru_medie = cv2.filter2D(imagine, -1, nucleu_medie)
filtru_gaussian = cv2.filter2D(imagine, -1, nucleu_gaussian)
filtru_laplace = cv2.filter2D(imagine, -1, nucleu_laplace)
filtru_highpass_4 = cv2.filter2D(imagine, -1, nucleu_highpass_4)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(imagine, cmap='gray')
plt.title('originala')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(filtru_medie, cmap='gray')
plt.title('mean (trece-jos)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(filtru_gaussian, cmap='gray')
plt.title('gaussian (trece-jos)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(filtru_laplace, cmap='gray')
plt.title('laplace (trece-sus)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(filtru_highpass_4, cmap='gray')
plt.title('high-pass (relatia 4)')
plt.axis('off')

plt.tight_layout()
plt.show()