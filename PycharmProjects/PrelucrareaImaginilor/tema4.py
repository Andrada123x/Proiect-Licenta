import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("forme.png", cv2.IMREAD_GRAYSCALE)
_, A = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

B = np.ones((3, 3), np.uint8)


eroded = cv2.erode(A, B)
contour = A - eroded

ys, xs = np.where(A == 255)
p = (ys[0], xs[0])

X = np.zeros_like(A)
X[p] = 255

A_comp = cv2.bitwise_not(A)

while True:
    X_new = cv2.dilate(X, B)
    X_new = cv2.bitwise_and(X_new, A_comp)

    if np.array_equal(X_new, X):
        break

    X = X_new

filled_interior = X
contour_plus_interior = cv2.bitwise_or(contour, filled_interior)

fig, ax = plt.subplots(2, 2, figsize=(8, 8))

ax[0][0].imshow(A, cmap="gray")
ax[0][0].set_title("Original binara")
ax[0][0].axis("off")

ax[0][1].imshow(contour, cmap="gray")
ax[0][1].set_title("Contur")
ax[0][1].axis("off")

ax[1][0].imshow(filled_interior, cmap="gray")
ax[1][0].set_title("Interior umplut (X)")
ax[1][0].axis("off")

ax[1][1].imshow(contour_plus_interior, cmap="gray")
ax[1][1].set_title("Contur + interior")
ax[1][1].axis("off")

plt.show()
