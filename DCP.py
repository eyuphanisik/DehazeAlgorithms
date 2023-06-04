"""Fog removal in images using improved dark channel prior and contrast limited adaptive histogram equalization"""

import cv2
import numpy as np
import os
import math
import time



def mdcp(img): # en karanlık kanala median filtre uygulanarak yumuşatılır
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

    estimate = mdcp(estimate)
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

def recover_mdcp(img, atmosphericLight, transmissionMap, t0=0.1):
    result = np.zeros_like(img)

    for i in range(0,3):
        result[:,:,i] = (img[:,:,i] - atmosphericLight[i]) / cv2.max(transmissionMap,t0) + atmosphericLight[i]

    return result