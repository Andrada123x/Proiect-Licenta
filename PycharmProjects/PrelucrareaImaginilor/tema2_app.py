import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt



def global_threshold_auto(image, epsilon=0.5):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    min_i = np.min(np.where(hist > 0))
    max_i = np.max(np.where(hist > 0))
    T = (min_i + max_i) / 2

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

    _, binary = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
    return binary, T



class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PI – Tema 2 / Lab 3")
        self.geometry("1100x700")

        self.image_path = None
        self.img_color = None
        self.gray = None
        self.binary = None
        self.equalized = None
        self.threshold_value = None

        self.output_dir = None

        self.build_ui()

    def build_ui(self):
        self.left = tk.Frame(self)
        self.left.pack(side="left", fill="y", padx=20, pady=20)

        self.right = tk.Frame(self)
        self.right.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        tk.Label(self.left, text="Opțiuni", font=("Arial", 18, "bold")).pack(anchor="w")

        tk.Button(self.left, text="Upload imagine",
                  font=("Arial", 12), width=22,
                  command=self.load_image).pack(pady=10)

        tk.Button(self.left, text="Prag global automat",
                  font=("Arial", 12), width=22,
                  command=self.run_threshold).pack(pady=6)

        tk.Button(self.left, text="Egalizare histogramă",
                  font=("Arial", 12), width=22,
                  command=self.run_equalization).pack(pady=6)

        tk.Button(self.left, text="Afișează histograme",
                  font=("Arial", 12), width=22,
                  command=self.show_histograms).pack(pady=6)

        tk.Label(self.left, text="Rezultatele se salvează automat",
                 font=("Arial", 10)).pack(pady=20)

        tk.Label(self.right, text="Preview", font=("Arial", 16, "bold")).pack()
        self.preview_label = tk.Label(self.right)
        self.preview_label.pack()

        self.info = tk.Label(self.right, text="", font=("Consolas", 11), justify="left")
        self.info.pack(pady=10, anchor="w")

        self.tk_img = None



    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.bmp *.tiff")]
        )
        if not path:
            return

        self.image_path = path
        self.img_color = cv2.imread(path)
        if self.img_color is None:
            messagebox.showerror("Eroare", "Nu pot deschide imaginea.")
            return

        self.gray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)

        base_dir = os.path.dirname(path)
        self.output_dir = os.path.join(base_dir, "output_tema_lab_3")
        os.makedirs(self.output_dir, exist_ok=True)

        cv2.imwrite(os.path.join(self.output_dir, "imagine_originala.jpg"), self.img_color)

        self.show_preview(self.img_color, "Imagine originală")

    def run_threshold(self):
        if self.gray is None:
            messagebox.showwarning("Atenție", "Încarcă o imagine!")
            return

        self.binary, self.threshold_value = global_threshold_auto(self.gray)
        out_path = os.path.join(
            self.output_dir,
            f"imagine_binarizata_{int(self.threshold_value)}.jpg"
        )
        cv2.imwrite(out_path, self.binary)

        self.show_preview(self.binary,
                          f"Binarizare\nPrag = {self.threshold_value:.2f}")

    def run_equalization(self):
        if self.gray is None:
            messagebox.showwarning("Atenție", "Încarcă o imagine!")
            return

        self.equalized = cv2.equalizeHist(self.gray)
        cv2.imwrite(os.path.join(self.output_dir, "imagine_egalizata.jpg"),
                    self.equalized)

        self.show_preview(self.equalized, "Imagine egalizată")

    def show_histograms(self):
        if self.gray is None or self.equalized is None:
            messagebox.showwarning("Atenție", "Rulează egalizarea întâi!")
            return

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Histogramă originală")
        plt.hist(self.gray.ravel(), 256, [0, 256])

        plt.subplot(1, 2, 2)
        plt.title("Histogramă egalizată")
        plt.hist(self.equalized.ravel(), 256, [0, 256])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "histograme_comparate.png"))
        plt.show()

    def show_preview(self, img, text):
        if len(img.shape) == 2:
            im = Image.fromarray(img)
        else:
            im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        im.thumbnail((700, 500))
        self.tk_img = ImageTk.PhotoImage(im)
        self.preview_label.configure(image=self.tk_img)
        self.info.configure(text=text)


if __name__ == "__main__":
    App().mainloop()
