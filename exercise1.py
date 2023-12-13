#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 08:48:53 2023

@author: shymontsh
"""

import numpy as np
import matplotlib.pyplot as plt
from IP01_function import correlation, getMaximumCorrPoint

plt.close('all')

# %% Functions
def calStatistics(img):
    if np.ndim(img) > 1:
        row, col = np.shape(img)
        n = row * col

        sumall = 0
        sumallSquared = 0
        for r in range(row):
            for c in range(col):
                sumall += img[r,c]
                sumallSquared += img[r,c]**2
                
        mean = sumall/n
        variance = sumallSquared/n - mean**2
        
    else:
        n = len(img)
        
        sumall = sum(img) #,dtype=np.float64)
        
        mean = sumall/n
        variance = sum((img-mean)**2)/n
    
    sd = np.sqrt(variance)
    
    return [mean, variance, sd]


def calHistogram(img, num_bit):
    row, col = np.shape(img)

    # define empty array of size image
    histArr = np.zeros(2**num_bit, dtype=int)
    for r in range(row):
        for c in range(col):
            histArr[img[r,c]] += 1
            
    return histArr


def calCovariance(img1, img2):
    if np.ndim(img1) > 1:
        row, col = np.shape(img1)
        n = row * col
        mean_img1 = sum([sum(i) for i in img1])/np.size(img1)
        mean_img2 = sum([sum(i) for i in img2])/np.size(img2)
        
    else:
        n = len(img1)
        mean_img1 = sum(img1) / n
        mean_img2 = sum(img2) / n
    
    # # use functions to speed up calculation of two matrices
    # multiply_img1img2 = np.multiply(img1, img2)
    # sum_img1img2 = np.sum(multiply_img1img2)
    # cov = (sum_img1img2 / (row*col)) - (mean_img1*mean_img2)
    
    cov = sum((img1[i] - mean_img1) * (img2[i] - mean_img2) for i in range(n)) / (n - 1)
    # cov_img1img2 = np.zeros((1,), dtype=np.float64)
    # for r in range(row):
    #     for c in range(col):
    #         cov_img1img2 += img1[r,c]*img2[r,c]
    
    return cov


def calCorrelation(img1, img2):
    [mean_img1, var_img1, sd_img1] = calStatistics(img1)
    [mean_img2, var_img2, sd_img2] = calStatistics(img2)
    
    sigma_img1img2 = calCovariance(img1, img2) * np.size(img1) # to compensate back n
    sigma_img1 = sum((img1 - mean_img1)**2)
    sigma_img2 = sum((img2 - mean_img2)**2)
    
    
    return sigma_img1img2 / np.sqrt(sigma_img1 * sigma_img2)
    
def templateSearch(img, temp):
    
    imgCorr = correlation(img, temp)
    [r, c] = getMaximumCorrPoint(imgCorr)
    
    return [imgCorr, r, c]
    

# %% Read input image
img = plt.imread("images/image.bmp")
img_a = plt.imread("images/image_a.bmp")
img_b = plt.imread("images/image_b.bmp")
img_c = plt.imread("images/image_c.bmp")
img_d = plt.imread("images/image_d.bmp")

num_bits = 8

# visualise input images
fig, axs = plt.subplots(2,2)
axs[0,0].set_title('img A')
axs[0,0].imshow(img_a, cmap='gray', vmin = 0, vmax = 2**num_bits-1) # to visualise in grayscale
axs[0,1].set_title('img B')
axs[0,1].imshow(img_b, cmap='gray', vmin = 0, vmax = 2**num_bits-1)
axs[1,0].set_title('img C')
axs[1,0].imshow(img_c, cmap='gray', vmin = 0, vmax = 2**num_bits-1)
axs[1,1].set_title('img D')
axs[1,1].imshow(img_d, cmap='gray', vmin = 0, vmax = 2**num_bits-1)
plt.tight_layout()
plt.savefig('images/img_inputs.png')

# %% Calculation of statistics
meanVal = np.mean(img)
varVal = np.var(img)
sdVal = np.std(img)
print('Python: mean = {}, variance = {}, SD = {}'.format(meanVal, varVal, sdVal))

[ownMeanVal, ownVarVal, ownSdVal] = calStatistics(img)
print('Own: mean = {}, variance = {}, SD = {}'.format(ownMeanVal, ownVarVal, ownSdVal))

h = calHistogram(img, 8)
# calculate mean using histogram method (sanity check)
mu_h = sum(h[i] * i for i in range (2**num_bits)) / np.size(img)
sigma2_h = sum(h[i] * (i - mu_h)**2 for i in range (2**num_bits)) / np.size(img)
print('Hist: mean = {}, variance = {}, SD = {}'.format(mu_h, sigma2_h, np.sqrt(sigma2_h)))


# %% Calculation of Cov and Corr
covVal = np.cov(img_a.flatten(),img_a.flatten()) # covariance of flattened arrays -> [[AA, AB],[BA, BB]]
corrVal = np.corrcoef(img_a.flatten(),img_a.flatten()) # correlation of flattened arrays -> [[AA, AB],[BA, BB]]
print('Python: Covariance = {},\n Correlation = {}'.format(covVal[0][1], corrVal[0][1])) # take AB or BA for cov betwn imgs

ownCovVal = calCovariance(img_a.flatten(),img_a.flatten()) # single value output
ownCorrVal = calCorrelation(img_a.flatten(),img_a.flatten()) # single value output
print('Own: Covariance = {},\n Correlation = {}'.format(ownCovVal, ownCorrVal))

fig, axes = plt.subplots(1,2)
axes[0].imshow(img, cmap='gray', vmin = 0, vmax = 2**num_bits-1)
axes[0].set_title("Original Img")
axes[1].stem(h, markerfmt = " ")
axes[1].set_title("Hist of original img")
plt.tight_layout()
plt.savefig('images/original_img.png')


# %% Test on images a, b, c, d
# img a
[imgA_MeanVal, imgA_VarVal, imgA_SdVal] = calStatistics(img_a)
h_a = calHistogram(img_a, num_bits)

# img b
[imgB_MeanVal, imgB_VarVal, imgB_SdVal] = calStatistics(img_b)
h_b = calHistogram(img_b, num_bits)

# img c
[imgC_MeanVal, imgC_VarVal, imgC_SdVal] = calStatistics(img_c)
h_c = calHistogram(img_c, num_bits)

# img d
[imgD_MeanVal, imgD_VarVal, imgD_SdVal] = calStatistics(img_d)
h_d = calHistogram(img_d, num_bits)

print('Image mean: A = {}, B = {}, C = {}, D = {}'.format(imgA_MeanVal, imgB_MeanVal, imgC_MeanVal, imgD_MeanVal))
print('Image var: A = {}, B = {}, C = {}, D = {}'.format(imgA_VarVal, imgB_VarVal, imgC_VarVal, imgD_VarVal))
print('Image sd: A = {}, B = {}, C = {}, D = {}'.format(imgA_SdVal, imgB_SdVal, imgC_SdVal, imgD_SdVal))

# plot histogram
fig, axes = plt.subplots(2,2)
 # draws lines perpendicular to a baseline at each location *locs* from the baseline to *heads*, and places a marker there
axes[0,0].stem(h_a, markerfmt = " ")
axes[0,0].set_title("img A")
axes[0,1].stem(h_b, markerfmt = " ")
axes[0,1].set_title("img B")
axes[1,0].stem(h_c, markerfmt = " ")
axes[1,0].set_title("img C")
axes[1,1].stem(h_d, markerfmt = " ")
axes[1,1].set_title("img D")

fig.suptitle("Histogram")
fig.supxlabel("bins")
fig.supylabel("count points")
plt.tight_layout()
plt.savefig('images/img_hists.png')

# # Compare images (any 2)
imgA_imgB_CovVal = calCovariance(img_a.flatten(),img_b.flatten())
imgA_imgB_CorrVal = calCorrelation(img_a.flatten(),img_b.flatten())
print('Cov(img A,img B) = {}, Corr(img A,img B) = {}'.format(imgA_imgB_CovVal, imgA_imgB_CorrVal))

imgA_imgC_CovVal = calCovariance(img_a.flatten(),img_c.flatten())
imgA_imgC_CorrVal = calCorrelation(img_a.flatten(),img_c.flatten())
print('Cov(img A,img C) = {}, Corr(img A,img C) = {}'.format(imgA_imgC_CovVal, imgA_imgC_CorrVal))

imgA_imgD_CovVal = calCovariance(img_a.flatten(),img_d.flatten())
imgA_imgD_CorrVal = calCorrelation(img_a.flatten(),img_d.flatten())
print('Cov(img A,img D) = {}, Corr(img A,img D) = {}'.format(imgA_imgD_CovVal, imgA_imgD_CorrVal))

imgB_imgC_CovVal = calCovariance(img_b.flatten(),img_c.flatten())
imgB_imgC_CorrVal = calCorrelation(img_b.flatten(),img_c.flatten())
print('Cov(img B,img C) = {}, Corr(img B,img C) = {}'.format(imgB_imgC_CovVal, imgB_imgC_CorrVal))

imgB_imgD_CovVal = calCovariance(img_b.flatten(),img_d.flatten())
imgB_imgD_CorrVal = calCorrelation(img_b.flatten(),img_d.flatten())
print('Cov(img B,img D) = {}, Corr(img B,img D) = {}'.format(imgB_imgD_CovVal, imgB_imgD_CorrVal))

imgC_imgD_CovVal = calCovariance(img_c.flatten(),img_d.flatten())
imgC_imgD_CorrVal = calCorrelation(img_c.flatten(),img_d.flatten())
print('Cov(img C,img D) = {}, Corr(img C,img D) = {}'.format(imgC_imgD_CovVal, imgC_imgD_CorrVal))

# # Compare images (original img vs. A/B/C/D)
img_imgA_CovVal = calCovariance(img.flatten(),img_a.flatten())
img_imgA_CorrVal = calCorrelation(img.flatten(),img_a.flatten())
print('Cov(img,img A) = {}, Corr(img,img A) = {}'.format(img_imgA_CovVal, img_imgA_CorrVal))

img_imgB_CovVal = calCovariance(img.flatten(),img_b.flatten())
img_imgB_CorrVal = calCorrelation(img.flatten(),img_b.flatten())
print('Cov(img,img B) = {}, Corr(img,img B) = {}'.format(img_imgB_CovVal, img_imgB_CorrVal))

img_imgC_CovVal = calCovariance(img.flatten(),img_c.flatten())
img_imgC_CorrVal = calCorrelation(img.flatten(),img_c.flatten())
print('Cov(img,img C) = {}, Corr(img,img C) = {}'.format(img_imgC_CovVal, img_imgC_CorrVal))

img_imgD_CovVal = calCovariance(img.flatten(),img_d.flatten())
img_imgD_CorrVal = calCorrelation(img.flatten(),img_d.flatten())
print('Cov(img,img D) = {}, Corr(img,img D) = {}'.format(img_imgD_CovVal, img_imgD_CorrVal))


# %% Template search
img_query = plt.imread("images/query.bmp")
temp_a = plt.imread("images/templateA.bmp")
temp_g = plt.imread("images/templateG.bmp")
temp_p = plt.imread("images/templateP.bmp")
temp_v = plt.imread("images/templateV.bmp")

# visualise query image
fig, ax = plt.subplots()
ax.set_title('Query image')
ax.imshow(img_query,cmap = plt.get_cmap("gray"))
plt.savefig('images/query_img.png')

# visualise template images
fig, axs = plt.subplots(2,2)
axs[0,0].set_title('temp A')
axs[0,0].imshow(temp_a,cmap = plt.get_cmap("gray"))
axs[0,1].set_title('temp G')
axs[0,1].imshow(temp_g,cmap = plt.get_cmap("gray"))
axs[1,0].set_title('temp P')
axs[1,0].imshow(temp_p,cmap = plt.get_cmap("gray"))
axs[1,1].set_title('temp V')
axs[1,1].imshow(temp_v,cmap = plt.get_cmap("gray"))
plt.tight_layout()
plt.savefig('images/img_templates.png')

# result most likly position template search 
corrImg_A, r_A, c_A = templateSearch(img_query, temp_a)    
corrImg_G, r_G, c_G = templateSearch(img_query, temp_g)
corrImg_P, r_P, c_P = templateSearch(img_query, temp_p)
corrImg_V, r_V, c_V = templateSearch(img_query, temp_v)

fig, axes = plt.subplots(2,2)
fig.suptitle("Correlation output")
axes[0,0].imshow(corrImg_A, cmap = plt.get_cmap("gray"))
axes[0,0].set_title("Img A")
axes[0,1].imshow(corrImg_G, cmap = plt.get_cmap("gray"))
axes[0,1].set_title("Img G")
axes[1,0].imshow(corrImg_P, cmap = plt.get_cmap("gray"))
axes[1,0].set_title("Img P")
axes[1,1].imshow(corrImg_V, cmap = plt.get_cmap("gray"))
axes[1,1].set_title("Img V")
plt.tight_layout()
plt.savefig('images/img_corr_output.png')

# plot result template search
fig, axes = plt.subplots(2,2)
axes[0,0].imshow(img_query, cmap = plt.get_cmap("gray"))
axes[0,0].scatter(x = [c_A], y = [r_A], c='r', s=10)
axes[0,0].set_title("Correlation A  " + str(c_A) + ', ' + str(r_A) )

axes[0,1].imshow(img_query,cmap = plt.get_cmap("gray"))
axes[0,1].scatter(x = [c_G], y = [r_G], c='r', s=10)
axes[0,1].set_title("Correlation G  " + str(c_G) + ', ' + str(r_G) )

axes[1,0].imshow(img_query,cmap = plt.get_cmap("gray"))
axes[1,0].scatter(x = [c_P], y = [r_P], c='r', s=10)
axes[1,0].set_title("Correlation P  " + str(c_P) + ', ' + str(r_P) )

axes[1,1].imshow(img_query,cmap = plt.get_cmap("gray"))
axes[1,1].scatter(x = [c_V], y = [r_V], c = 'r', s=10)
axes[1,1].set_title("Correlation V:  " + str(c_V) + ', ' + str(r_V) )
plt.tight_layout()
plt.savefig('images/img_template_match.png')
