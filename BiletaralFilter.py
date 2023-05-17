import numpy as np
import cv2
import os

def histogramEqualization(img):
    imgYuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    imgYuv[:,:,0] = cv2.equalizeHist(imgYuv[:,:,0])
    return cv2.cvtColor(imgYuv,cv2.COLOR_YUV2BGR)

def atmosferikIsikTahmini(img, beta):
    img = img / 255
    b,g,r = cv2.split(img)
    minimum = cv2.min(b,g)
    darkChannel = cv2.min(minimum,r) * beta
   
    return darkChannel

def atmosferikIsiginIyilestirilmesi(img):
    imgB = cv2.bilateralFilter((img * 255).astype('uint8'), 9,50,50)
    return imgB

def recover(img, atmosphericLight, i_inf = [1]):
    b,g,r = cv2.split(img/255)
    atm = atmosphericLight /255

    b = ((b - atm) / (1.0 - atm/ i_inf))
    g = ((g - atm) / (1.0 - atm/ i_inf))
    r = ((r - atm) / (1.0 - atm/ i_inf))
    result = cv2.merge((b, g, r))
    return (np.clip(result, 0, 1)*255).astype('uint8')

def stretch(img):
    result = np.zeros_like(img)

    for i in range(3):
        result[:,:,i] = (img[:,:,i] - np.min(img[:,:,i])) * (255 / np.max(img[:,:,i]) - np.min(img[:,:,i]) )  
    
    return result

"""img = cv2.imread("DCP\\1.jpg")
img = cv2.resize(img, [640,480])
he = histogramEqualization(img)
a = atmosferikIsikTahmini(img, 0.9)
biletaral = atmosferikIsiginIyilestirilmesi(a)
result = recover(he,biletaral)
b,g,r = cv2.split(result)
ab =stretch(result)

cv2.imshow("a", ab)
cv2.waitKey(0)"""

folder_path = 'D:\Windows10\Desktop\Database\\O-HAZY'
    
files = os.path.join(folder_path, "hazy")

for i in range(1,46):
    img = folder_path + "\hazy/" + str(i) + "_outdoor_hazy.jpg"
    src = cv2.imread(img)
    img = cv2.resize(src, [480,640])
    he = histogramEqualization(img)
    a = atmosferikIsikTahmini(img, 0.9)
    biletaral = atmosferikIsiginIyilestirilmesi(a)
    result = recover(he,biletaral)
    b,g,r = cv2.split(result)
    
    ab =stretch(result)
    
    cv2.imshow("a", ab)
    cv2.waitKey(0)