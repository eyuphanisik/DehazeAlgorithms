import utils
import imresize
import cv2
import numpy as np
import math

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