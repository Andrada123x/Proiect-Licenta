import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


image_path = r"C:\Users\User1\Desktop\Flower.jpg"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Imaginea nu a fost găsită la calea specificată.")


_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)


contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if not contours:
    raise ValueError("Nu a fost detectat niciun obiect.")

contour = max(contours, key=cv2.contourArea)

#
freeman_directions = {
    (1, 0): 0,
    (1, -1): 1,
    (0, -1): 2,
    (-1, -1): 3,
    (-1, 0): 4,
    (-1, 1): 5,
    (0, 1): 6,
    (1, 1): 7
}

chain_code = []
for i in range(1, len(contour)):
    x_prev, y_prev = contour[i - 1][0]
    x_curr, y_curr = contour[i][0]
    dx, dy = x_curr - x_prev, y_curr - y_prev
    d = freeman_directions.get((dx, dy))
    if d is not None:
        chain_code.append(d)

print("Lungime cod înlănțuit:", len(chain_code))
print("Primele 50 valori:", chain_code[:50])


vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(vis, [contour], -1, (0, 0, 255), 2)

# Folderul unde se află fișierul .py
script_dir = os.path.dirname(os.path.abspath(__file__))

output_path = os.path.join(script_dir, "contur_obiect.png")

# Salvare imagine
cv2.imwrite(output_path, vis)

print(f"Imaginea cu contur a fost salvată la:\n{output_path}")


plt.figure(figsize=(6, 6))
plt.imshow(vis[..., ::-1])
plt.title("Conturul obiectului (salvat)")
plt.axis("off")
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(binary, cmap="gray")
plt.title("Imagine binară folosită")
plt.axis("off")
plt.show()