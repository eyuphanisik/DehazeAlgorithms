"""Fog removal in images using improved dark channel prior and contrast limited adaptive histogram equalization"""

import cv2
import numpy as np
import os
import math
import time
import utils
import imresize

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def SSIM(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def MSE(image1, image2):
    return np.square(image1 - image2).mean()

def contrastGain(image1, image2):
    image1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    image2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # Görüntüler arasındaki farkı hesaplayın
    diff = cv2.absdiff(image1, image2)

    # Farkın ortalama değerini hesaplayın
    mean_diff = np.mean(diff)

    return mean_diff

def compare_images_ciede2000(image1, image2):
    
    gray1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    lab1 = cv2.cvtColor(gray1, cv2.COLOR_RGB2LAB)
    gray2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    lab2 = cv2.cvtColor(gray2, cv2.COLOR_RGB2LAB)

    
    # Calculate CIEDE2000 color difference
    ciede2000 = cv2.compareHist(cv2.calcHist([lab1], [0, 1], None, [32, 32], [0, 256, 0, 256]), 
                                cv2.calcHist([lab2], [0, 1], None, [32, 32], [0, 256, 0, 256]), 
                                cv2.HISTCMP_BHATTACHARYYA)
    
    # Return CIEDE2000 color difference score
    return ciede2000

def brisque(image):
    y_mscn = utils.compute_image_mscn_transform(image)
    half_scale = imresize.imresize(image, scalar_scale = 0.5, method = 'bicubic')
    y_half_mscn = utils.compute_image_mscn_transform(half_scale)
    feats_full = utils.extract_subband_feats(y_mscn)
    feats_half = utils.extract_subband_feats(y_half_mscn)
    return np.concatenate((feats_full, feats_half))

def doygunluk(image1, image2):
    gray1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    gray_img1 = cv2.cvtColor(gray1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    gray_img2 = cv2.cvtColor(gray2, cv2.COLOR_RGB2GRAY)

    # Resimlerin histogramlarını eşitleyin
    equalized_img1 = cv2.equalizeHist(gray_img1)
    equalized_img2 = cv2.equalizeHist(gray_img2)

    # İki resmin histogramlarının ortalama değerlerini hesaplayın
    mean1 = cv2.mean(equalized_img1)[0]
    mean2 = cv2.mean(equalized_img2)[0]

    # İki resim arasındaki doygunluk farkını hesaplayın
    contrast = abs(mean1 - mean2)

    return contrast

def AMBE(image1, image2):
    
    # Resimleri Gri tonlamalı görüntüye dönüştürün
    gray1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    gray_img1 = cv2.cvtColor(gray1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    gray_img2 = cv2.cvtColor(gray2, cv2.COLOR_RGB2GRAY)

   # Calculate mean brightness values for each image
    mean_brightness_img1 = np.mean(gray_img1)
    mean_brightness_img2 = np.mean(gray_img2)

    # Calculate absolute mean brightness error (AMBE)
    ambe = abs(mean_brightness_img1 - mean_brightness_img2)
    return ambe

def dcp(img): # en karanlık kanala median filtre uygulanarak yumuşatılır
    b,g,r = cv2.split(img)
    minimum = cv2.min(b,g)
    minimum = cv2.min(minimum,r)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    dark = cv2.erode(minimum,kernel)
    return dark

def atmosphericLightEstimate(img, dark_channel):
    [h,w] = img.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1)) # resimdeki piksel sayısını 1000e böler ve sonucu en yakın tam sayıya yuvarlar
    darkvec = dark_channel.reshape(imsz) # dcp sonucunu tek boyuta çevirir
    imvec = img.reshape(imsz,3) # resimdeki her kanalı tek boyuta çevirir

    indices = darkvec.argsort() # piksel değerlerini sıralar
    indices = indices[imsz-numpx::] #en yüksek 1000 pikseli alır

    atmsum = np.zeros([1,3]) #3 tane boş elemanlı dizi oluşturur
    for ind in range(1,numpx): # 1000e bölünmüş piksel sayısı kadar döner
       atmsum = atmsum + imvec[indices[ind]] # her kanaldaki en yüksek pikselleri toplar

    A = atmsum / numpx # çıkan sonucu resimdeki piksel sayısının 1000e bölünmüş haline böler
    return A[0]

def transmissionMapEstimate(img, atmosphericLight,omega=0.95): # 1 - omega *  dcp
    estimate = np.zeros_like(img)
    # print(img.shape)
    for i in range(0,3):
        estimate[:,:,i] = img[:,:,i] / atmosphericLight[i]

    estimate = dcp(estimate)
    estimate = 1 - omega *estimate
    return estimate

def guidedFilter(img, transmissionMap, r=64, eps=0.0001): # iletim haritasını iyileştirmek için kullanılır
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
ssimTotal = 0
psnrTotal = 0
mseTotal = 0
doygunlukTotal = 0
contrastGainTotal = 0
ambeTotal = 0
ciede2000Total = 0
brisqueTotal = 0
print(len(os.listdir(files)))
start_time = time.time()
for k in range(1,46):
    img = folder_path + "\hazy/" + str(k) + "_outdoor_hazy.jpg"
    img = cv2.imread(img)
    img = cv2.resize(img, [480,640]) # hepsinin boyutu aynı olsun diye resize ettik
    i = img / 255 #normalization

    d = dcp(i)
    a = atmosphericLightEstimate(i, d)

    t = transmissionMapEstimate(i, a)
    
    g = guidedFilter(img, t)
    
    r = recover(i, a, g)
    cv2.imshow("i",r)
    cv2.waitKey(0)

    print(folder_path)
    orj = cv2.imread(folder_path + "\GT/"+ str(k) + "_outdoor_GT.jpg")
    
    orj = cv2.resize(orj, [480, 640])
    sissiz = r*255
    sissiz = sissiz.astype(np.uint8)
    
    img_gray = cv2.cvtColor(sissiz, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype(np.float64)
    
    feats = brisque(img_gray)
    np.set_printoptions(precision=4)
    bris = np.sum(feats)
    brisqueTotal += bris
    
    ssim = SSIM(orj, sissiz)
    print(f"SSIM: {ssim}")
    psnr = PSNR(orj, sissiz)
    print(f"PSNR: {psnr}")
    mse = MSE(orj, sissiz)
    print(f"MSE: {mse} {k}")
    contrast = contrastGain(img, sissiz)
    print(f"contrast: {contrast} {k}")
    doygun = doygunluk(orj, sissiz)
    print(f"doygunluk: {doygun} {k}")
    ambe = AMBE(orj, sissiz)
    print(f"AMBE: {ambe} {k}")
    ciede2000 = compare_images_ciede2000(orj, sissiz)
    print(f"ciede2000: {ciede2000} {k}")
    print(f"brisque: {bris} {k}")
    mseTotal += mse
    psnrTotal += psnr
    ssimTotal += ssim
    contrastGainTotal += contrast
    doygunlukTotal += doygun
    ambeTotal += ambe
    ciede2000Total += ciede2000
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("İşlem {} saniye sürdü.".format(elapsed_time/len(os.listdir(files))))
    #print(len(files))
print("-----------------------")
print(f"SSIM: {ssimTotal / len(os.listdir(files))}")
print(f"PSNR: {psnrTotal / len(os.listdir(files))}")
print(f"MSE: {mseTotal / len(os.listdir(files))}")
print(f"Kontrast: {contrastGainTotal / len(os.listdir(files))}")
print(f"Doygunluk: {doygunlukTotal / len(os.listdir(files))}")
print(f"AMBE: {ambeTotal / len(os.listdir(files))}")
print(f"CIEDE2000: {ciede2000Total / len(os.listdir(files))}")
print(f"BRISQUE: {brisqueTotal / len(os.listdir(files))}")