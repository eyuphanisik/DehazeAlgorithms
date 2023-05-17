import cv2
import numpy as np
import os

def dcp(img):
    b,g,r = cv2.split(img)
    minimum = cv2.min(b,g)
    return cv2.min(minimum,r)

def atmosphericLightEstimate(img, dark_channel):
    h, w = img.shape[:2]
    # En yüksek %0.1'lik kısmı al
    top_pixels = int(h * w * 0.001)
    top_dark = dark_channel.ravel().argsort()[-top_pixels:]

    # Atmosferik ışığı seç
    A = np.max(img.reshape((h * w, 3))[top_dark], axis=0)
    return A
def transmissionMapEstimate(img, atmosphericLight,omega=0.95):
    estimate = np.zeros_like(img)
    # print(img.shape)
    for i in range(0,3):
        estimate[:,:,i] = img[:,:,i] / atmosphericLight[i]

    estimate = dcp(estimate)
    estimate = 1 - omega *estimate
    return estimate

def guidedFilter(img, transmissionMap, r=64, eps=0.0001):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    meanIg = cv2.boxFilter(imgGray,cv2.CV_64F,(r,r))
    meanIt = cv2.boxFilter(transmissionMap, cv2.CV_64F,(r,r))
    meanIgt = cv2.boxFilter(imgGray*transmissionMap,cv2.CV_64F,(r,r))
    covIgt = meanIgt - meanIg*meanIt

    meanII = cv2.boxFilter(imgGray*imgGray,cv2.CV_64F,(r,r))
    varI   = meanII - meanIg*meanIg

    a = covIgt/(varI + eps)
    b = meanIt - a*meanIg

    meanA = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    meanB = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = meanA*imgGray + meanB
    return q

def CLAHE(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # LAB görüntüsünü ayrı ayrı kanallara böl
    l, a, b = cv2.split(lab)

    # L kanalı için CLAHE uygula
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)

    # A kanalı için CLAHE uygula
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    a_clahe = clahe.apply(a)

    # B kanalı için CLAHE uygula
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    b_clahe = clahe.apply(b)

    # Kanalları tekrar birleştir
    lab_clahe = cv2.merge((l_clahe, a_clahe, b_clahe))

    # LAB görüntüsünü tekrar BGR renk uzayına dönüştür
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def recover(img, atmosphericLight, transmissionMap, t0=0.1):
    result = np.zeros_like(img)

    for i in range(0,3):
        result[:,:,i] = (img[:,:,i] - atmosphericLight[i]) / cv2.max(transmissionMap,t0) + atmosphericLight[i]

    return result
                         
"""img = cv2.imread("D:\Windows10\Desktop\projeler\dehaze\DCP\\1_outdoor_hazy.jpg")
print(img.shape)
img = cv2.resize(img, [480,640])
i = img / 255
# cv2.imshow("a",img)
# cv2.waitKey(0)
d = dcp(i)

a = atmosphericLightEstimate(i, d)

t = transmissionMapEstimate(i, a)
g = guidedFilter(img, t)

r = recover(i, a, g)

cv2.imshow("ab",r)
cv2.waitKey(0)"""


folder_path = 'D:\Windows10\Desktop\Database\\O-HAZY'
    
files = os.path.join(folder_path, "hazy")

for i in range(1,46):
    img = folder_path + "\hazy/" + str(i) + "_outdoor_hazy.jpg"
    img = cv2.imread(img)
    img = cv2.resize(img, [480,640])
    i = img / 255
    # cv2.imshow("a",img)
    # cv2.waitKey(0)
    d = dcp(i)

    a = atmosphericLightEstimate(i, d)

    t = transmissionMapEstimate(i, a)
    g = guidedFilter(img, t)

    r = recover(i, a, g)

    cv2.imshow("i",r)
    cv2.waitKey(0)