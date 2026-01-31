import numpy as np
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt


# ALGORTIM 1 – BFS (Traversare în lățime)

def bfs_labeling(binary_img):
    h, w = binary_img.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 1

    directions = [(1,0), (-1,0), (0,1), (0,-1)]  # 4 conexiuni

    for i in range(h):
        for j in range(w):
            if binary_img[i, j] == 0 and labels[i, j] == 0:
                queue = deque()
                queue.append((i, j))
                labels[i, j] = current_label

                while queue:
                    x, y = queue.popleft()

                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if binary_img[nx, ny] == 0 and labels[nx, ny] == 0:
                                labels[nx, ny] = current_label
                                queue.append((nx, ny))

                current_label += 1

    return labels



# ALGORTIM 2 – Two-pass (Clasa de echivalențe)

def two_pass_labeling(binary_img):
    h, w = binary_img.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 1

    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra


    for i in range(h):
        for j in range(w):
            if binary_img[i, j] == 0:

                neighbors = []

                if i > 0 and labels[i-1, j] > 0:
                    neighbors.append(labels[i-1, j])
                if j > 0 and labels[i, j-1] > 0:
                    neighbors.append(labels[i, j-1])

                if len(neighbors) == 0:
                    labels[i, j] = label
                    parent[label] = label
                    label += 1
                else:
                    m = min(neighbors)
                    labels[i, j] = m
                    for n in neighbors:
                        union(m, n)


    for i in range(h):
        for j in range(w):
            if labels[i, j] > 0:
                labels[i, j] = find(labels[i, j])

    return labels



def main():
    # Încarcă imaginea
    img = Image.open('forme.png').convert('L')
    binary = np.array(img)

    # Convertim în imagine binară (0 = obiect, 255 = fundal)
    binary = np.where(binary > 128, 255, 0)

    # Aplicăm algoritmii
    labels_bfs = bfs_labeling(binary)
    labels_two_pass = two_pass_labeling(binary)

    # Afișăm rezultatele
    print("Număr obiecte (Traversare în lățime):", labels_bfs.max())
    print("Număr obiecte (Două treceri cu clase de echivalențe):", labels_two_pass.max())

    # Afișare vizuală
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(binary, cmap='gray')
    axs[0].set_title("Imagine binară")

    axs[1].imshow(labels_bfs, cmap='nipy_spectral')
    axs[1].set_title("Etichetare BFS")

    axs[2].imshow(labels_two_pass, cmap='nipy_spectral')
    axs[2].set_title("Două treceri cu clase de echivalențe")

    for ax in axs:
        ax.axis('off')

    plt.show()



if __name__ == "__main__":
    main()
