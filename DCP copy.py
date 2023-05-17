import cv2
import numpy as np

# Görüntüyü yükle
img = cv2.imread("D:\Windows10\Desktop\projeler\dehaze\DCP/1.jpg")

# Boyutu al
h, w = img.shape[:2]

# Dark channel prior hesapla
kernel_size = 15
dark_img = np.zeros((h, w))
for i in range(h):
    for j in range(w):
        patch = img[i:i+kernel_size, j:j+kernel_size]
        dark_img[i, j] = np.min(patch)

# En yüksek %0.1'lik kısmı al
top_pixels = int(h * w * 0.001)
top_dark = dark_img.ravel().argsort()[-top_pixels:]

# Atmosferik ışığı seç
A = np.max(img.reshape((h * w, 3))[top_dark], axis=0)

# Atmosferik ışık tahminini uygula
t = 0.1
J = np.zeros_like(img, dtype=np.float32)
for i in range(3):
    J[:,:,i] = (img[:,:,i].astype(np.float32) - A[i])/t + A[i]

# Sonuçları kaydet
cv2.imwrite("result.jpg", J)

                         
