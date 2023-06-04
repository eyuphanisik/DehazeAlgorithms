import os
import time
import cv2
import numpy as np
from DCP import mdcp, atmosphericLightEstimate, transmissionMapEstimate, guidedFilter, recover_mdcp
from BiletaralFilter import atmosferikIsiginIyilestirilmesi, atmosferikIsikTahmini, histogramEqualization, recover_bil, stretch
from comparisonParameters import brisque, AMBE, compare_images_ciede2000, contrastGain, SSIM, PSNR, MSE, doygunluk

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
    img = cv2.resize(img, [640, 480]) # hepsinin boyutu aynı olsun diye resize ettik
    i = img / 255 #normalization
    """MDCP Algoritması"""
    d = mdcp(i)
    a = atmosphericLightEstimate(i, d)
    t = transmissionMapEstimate(i, a)
    g = guidedFilter(img, t)
    result_mdcp = recover_mdcp(i, a, g)

    """Bileteral Filtre Algoritması"""
    he = histogramEqualization(img)
    a = atmosferikIsikTahmini(img, 0.9)
    biletaral = atmosferikIsiginIyilestirilmesi(a)
    result = recover_bil(he,biletaral)
    result_bil =stretch(result)

    orj = cv2.imread(folder_path + "\GT/"+ str(k) + "_outdoor_GT.jpg")
    
    orj = cv2.resize(orj, [600, 400]) #resimlerin ekrana sığması için tekrar resize edildi
    img = cv2.resize(img, [600, 400])
    result_mdcp = cv2.resize(result_mdcp, [600, 400])
    result_bil = cv2.resize(result_bil, [600, 400])
    
    orj =  orj/255
    img =  img/255
    sonuc_resim_ust = np.concatenate((orj, img), axis=1)
    sonuc_resim_alt = np.concatenate((result_bil, result_mdcp), axis=1)
    sonuc_resim = np.concatenate((sonuc_resim_ust, sonuc_resim_alt), axis=0)

    cv2.imshow("sisi giderilmis",sonuc_resim)
    cv2.waitKey(0)

    img_gray = cv2.cvtColor((result_mdcp*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype(np.float64)
    
    feats = brisque(img_gray)
    np.set_printoptions(precision=4)
    bris = np.sum(feats)
    brisqueTotal += bris
    
    ssim = SSIM(orj, result_mdcp)
    print(f"SSIM: {ssim}")
    psnr = PSNR(orj, result_mdcp)
    print(f"PSNR: {psnr}")
    mse = MSE(orj, result_mdcp)
    print(f"MSE: {mse} {k}")
    contrast = contrastGain(img, result_mdcp)
    print(f"contrast: {contrast} {k}")
    doygun = doygunluk(orj, result_mdcp)
    print(f"doygunluk: {doygun} {k}")
    ambe = AMBE(orj, result_mdcp)
    print(f"AMBE: {ambe} {k}")
    ciede2000 = compare_images_ciede2000(orj, result_mdcp)
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