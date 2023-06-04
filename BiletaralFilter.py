"""Single Image Fog Removal Using Bilateral Filter"""

import numpy as np
import cv2

def histogramEqualization(img): # histogram eşitleme
    imgYuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    imgYuv[:,:,0] = cv2.equalizeHist(imgYuv[:,:,0])
    return cv2.cvtColor(imgYuv,cv2.COLOR_YUV2BGR)

def atmosferikIsikTahmini(img, beta): #dcp
    img_c = img/255
    if img.shape[-1] >1:
        min_img = beta*(np.amin(img_c, axis = 2))
    else:
        min_img = beta*(img_c)
    #print (min_img.shape)
    return (min_img)

def atmosferikIsiginIyilestirilmesi(img): # bileteral filtre

    imgB = cv2.bilateralFilter(((img)*255).astype('uint8'), 9, 75, 75)
    return imgB

def recover_bil(img, atmosphericLight, i_inf = [1]): #formüle her kanalın ayrı ayrı resterasyonu yapılıp birleştirilir

    b,g,r = cv2.split(img/255)
    atm = atmosphericLight /255

    b = ((b - atm) / (1.0 - atm/ i_inf))
    g = ((g - atm) / (1.0 - atm/ i_inf))
    r = ((r - atm) / (1.0 - atm/ i_inf))
    result = cv2.merge((b, g, r))
    return (np.clip(result, 0, 1)*255).astype('uint8')

def stretch(img): #resmin post processing işlemi yapılıp çıktı elde edilir
    # R, G ve B kanallarını ayır
    r, g, b = cv2.split(img)

    # Histogram eşitleme uygula
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)

    # Eşitlenmiş kanalları birleştir
    image_eq = cv2.merge([r_eq, g_eq, b_eq]) 
    
    return image_eq/255