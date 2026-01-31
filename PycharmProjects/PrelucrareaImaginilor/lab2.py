import cv2
import os
import matplotlib.pyplot as plt

# Calea imaginii originale (modifică dacă e nevoie)
image_path = r'C:\Users\User1\Desktop\flower.jpg'

# Directorul unde vrei să salvezi imaginea (Desktop)
save_path = r'C:\Users\User1\Desktop'

# Citește imaginea
img = cv2.imread(image_path)

# Verifică dacă imaginea există
if img is None:
    print("Eroare: Imaginea nu există la calea specificată!")
    exit()

# Afișează imaginea originală
cv2.imshow('Imagine originala', img)
cv2.waitKey(0)

# Afișează conținutul Desktop-ului înainte de salvare
print("Înainte de salvare:")
print(os.listdir(save_path))

# Creează calea completă pentru fișierul salvat
filename = os.path.join(save_path, 'SavedFlower.jpg')

# Salvează imaginea pe Desktop
cv2.imwrite(filename, img)
print(f"Imagine salvată cu succes la: {filename}")

print("După salvare:")
print(os.listdir(save_path))

# Conversie la tonuri de gri
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagine in tonuri de gri', gray)
cv2.waitKey(0)

# Conversie în alb-negru
(thresh, BlackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Imagine alb-negru', BlackAndWhiteImage)
cv2.waitKey(0)

# Conversie în spațiul de culoare HSV
hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('Imagine HSV', hsvImage)
cv2.waitKey(0)

# Închide toate ferestrele OpenCV
cv2.destroyAllWindows()

# Histograma imaginii grayscale
plt.figure(figsize=(6,4))
plt.hist(gray.ravel(), 256, [0, 256])
plt.title("Histograma imaginii în tonuri de gri")
plt.xlabel("Intensitate")
plt.ylabel("Frecvență")
plt.show()

# Histograma imaginii color
color = ('b', 'g', 'r')
plt.figure(figsize=(6,4))
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.title("Histograma imaginii color")
plt.xlabel("Intensitate")
plt.ylabel("Frecvență")
plt.show()
