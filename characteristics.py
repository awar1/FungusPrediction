import os
import sys
import cv2 as cv
import numpy as np
import skimage.filters
import skimage.color
import skimage.io
from skimage import img_as_ubyte
from scipy.stats import kurtosis
import skimage.feature
from skimage import color
from PIL import ImageStat, Image, ImageCms

# data from training pictures were transformed into csv files for speed of algorithm
thresholdOtsu = np.loadtxt(("tables/otsu.csv"))
circles = np.loadtxt(("tables/circles.csv"))
density = np.loadtxt(("tables/density.csv"))
redStdDev = np.loadtxt(("tables/redStdDev.csv"))
greenStdDev = np.loadtxt(("tables/greenStdDev.csv"))
blueStdDev = np.loadtxt(("tables/blueStdDev.csv"))
kurRed = np.loadtxt(("tables/kurRed.csv"))
kurBlue = np.loadtxt(("tables/kurBlue.csv"))
kurGreen = np.loadtxt(("tables/kurGreen.csv"))
kurHue = np.loadtxt(("tables/kurHue.csv"))
kurSat = np.loadtxt(("tables/kurSat.csv"))
kurVal = np.loadtxt(("tables/kurVal.csv"))
kurLab = np.loadtxt(("tables/kurLab.csv"))
kurA = np.loadtxt(("tables/kurA.csv"))
kurB = np.loadtxt(("tables/kurB.csv"))
redMean = np.loadtxt(("tables/redMean.csv"))
greenMean = np.loadtxt(("tables/greenMean.csv"))
blueMean = np.loadtxt(("tables/blueMean.csv"))
avgR = np.loadtxt(("tables/avgr.csv"))
avgG = np.loadtxt(("tables/avgG.csv"))
avgB = np.loadtxt(("tables/avgG.csv"))
avgH = np.loadtxt(("tables/avgH.csv"))
avgS = np.loadtxt(("tables/avgS.csv"))
avgV = np.loadtxt(("tables/avgV.csv"))
avgL = np.loadtxt(("tables/avgL.csv"))
avgA = np.loadtxt(("tables/avgA.csv"))
avgBB = np.loadtxt(("tables/avgBB.csv"))
contrasttab1 = np.loadtxt(("tables/contrasttab1.csv"))
contrasttab2 = np.loadtxt(("tables/contrasttab2.csv"))
contrasttab3 = np.loadtxt(("tables/contrasttab3.csv"))
contrasttab4 = np.loadtxt(("tables/contrasttab4.csv"))
dissimilaritytab1 = np.loadtxt(("tables/dissimilaritytab1.csv"))
dissimilaritytab2 = np.loadtxt(("tables/dissimilaritytab2.csv"))
dissimilaritytab3 = np.loadtxt(("tables/dissimilaritytab3.csv"))
dissimilaritytab4 = np.loadtxt(("tables/dissimilaritytab4.csv"))
homogeneitytab1 = np.loadtxt(("tables/homogeneitytab1.csv"))
homogeneitytab2 = np.loadtxt(("tables/homogeneitytab2.csv"))
homogeneitytab3 =np.loadtxt(("tables/homogeneitytab3.csv"))
homogeneitytab4 = np.loadtxt(("tables/homogeneitytab4.csv"))
energytab1 = np.loadtxt(("tables/energytab1.csv"))
energytab2 = np.loadtxt(("tables/energytab2.csv"))
energytab3 = np.loadtxt(("tables/energytab3.csv"))
energytab4 = np.loadtxt(("tables/energytab4.csv"))
correlationtab1 = np.loadtxt(("tables/correlationtab1.csv"))
correlationtab2 = np.loadtxt(("tables/correlationtab2.csv"))
correlationtab3 = np.loadtxt(("tables/correlationtab3.csv"))
correlationtab4 = np.loadtxt(("tables/correlationtab4.csv"))
varianceRed = np.loadtxt(("tables/varianceRed.csv"))
varianceGreen = np.loadtxt(("tables/varianceGreen.csv"))
varianceBlue = np.loadtxt(("tables/varianceBlue.csv"))
varianceH = np.loadtxt(("tables/varianceH.csv"))
varianceS = np.loadtxt(("tables/varianceS.csv"))
varianceV = np.loadtxt(("tables/varianceV.csv"))
varianceL = np.loadtxt(("tables/varianceL.csv"))
varianceA = np.loadtxt(("tables/varianceA.csv"))
varianceB = np.loadtxt(("tables/varianceB.csv"))
skewnessRed = np.loadtxt(("tables/skewnessRed.csv"))
skewnessGreen = np.loadtxt(("tables/skewnessGreen.csv"))
skewnessBlue = np.loadtxt(("tables/skewnessBlue.csv"))
skewnessH = np.loadtxt(("tables/skewnessH.csv"))
skewnessS = np.loadtxt(("tables/skewnessS.csv"))
skewnessV = np.loadtxt(("tables/skewnessV.csv"))
skewnessL = np.loadtxt(("tables/skewnessL.csv"))
skewnessA = np.loadtxt(("tables/skewnessA.csv"))
skewnessB = np.loadtxt(("tables/skewnessB.csv"))
hueStdDev = np.loadtxt(("tables/hueStdDev.csv"))
saturationStdDev = np.loadtxt(("tables/saturationStdDev.csv"))
valueStdDev = np.loadtxt(("tables/valueStdDev.csv"))
lightnessStdDev = np.loadtxt(("tables/lightnessStdDev.csv"))
aStdDev = np.loadtxt(("tables/aStdDev.csv"))
bStdDev = np.loadtxt(("tables/bStdDev.csv"))
edgeDensity = np.loadtxt(("tables/edgeDensity.csv"))
edgeBmean = np.loadtxt(("tables/edgeBmean.csv"))
edgeGmean = np.loadtxt(("tables/edgeGmean.csv"))
edgeRmean = np.loadtxt(("tables/edgeRmean.csv"))
edgeBstdDev = np.loadtxt(("tables/edgeBstdDev.csv"))
edgeGstdDev = np.loadtxt(("tables/edgeGstdDev.csv"))
edgeRstdDev = np.loadtxt(("tables/edgeRstdDev.csv"))


result = []
def isgray(img):
    """checks if picture is gray scale"""
    if len(img.shape) < 3: return True
    if img.shape[2]  == 1: return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False

def StandardDeviationHSV(img):
    """appends result with standard deviation of hsv colors on picture"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    a, std = cv.meanStdDev(hsv)
    result.append(std[0])
    result.append(std[1])
    result.append(std[2])

def StandardDeviationLAB(img):
    """appends result with standard deviation of lab colors on picture"""
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    a, std = cv.meanStdDev(lab)
    result.append(std[0])
    result.append(std[1])
    result.append(std[2])

def Edges(img):
    edges = cv.Canny(image=img, threshold1=100, threshold2=200)
    density = np.sum(edges != 0)
    result.append(density)
    gray = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    masked = cv.bitwise_and(img, gray, mask=None)
    b = masked[:, :, 0]
    g = masked[:, :, 1]
    r = masked[:, :, 2]
    b_mean = np.average(b[b != 0])
    g_mean = np.average(g[g != 0])
    r_mean = np.average(r[r != 0])
    b_stddev = np.std(b[b != 0])
    g_stddev = np.std(g[g != 0])
    r_stddev = np.std(r[r != 0])
    result.append(b_mean)
    result.append(g_mean)
    result.append(r_mean)
    result.append(b_stddev)
    result.append(g_stddev)
    result.append(r_stddev)

def circlesKey(img):
    l = 100 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    rows = gray.shape[0]
    circlesT = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / l,
                              param1=100, param2=30,
                              minRadius=1, maxRadius=30)
    if circlesT is not None:
        result.append(len(circlesT[0]))
    else:
        result.append(0)
        
def thresholdOtsuKey(img):
    """appends the threshold value of otsu thresholding on a picture"""
    if(isgray(img)):
        result.append(0)
    else:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(gray,5)
        ret, th = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        result.append(ret)

def meanStandardDeviationKey(img):
    """appends mean and standard deviation of rgb colors on a picture"""
    mean, std = cv.meanStdDev(img)
    result.append(mean[0])
    result.append(mean[1])
    result.append(mean[2])
    result.append(std[0])
    result.append(std[1])
    result.append(std[2])

def densityKey(img):
    b=0.2
    skimageImg = img_as_ubyte(img)
    gray = skimage.color.rgb2gray(skimageImg)
    blurred_image = skimage.filters.gaussian(gray, sigma=1.0)
    mask = blurred_image < b
    labeled_image, count = skimage.measure.label(mask, return_num=True)
    result.append(count)

def kurtosisRGBKey(img):
    src = img
    #red
    red_channel = src[:,:,2]
    red_img = np.zeros(src.shape)
    red_img[:,:,2] = red_channel
    result.append(kurtosis(red_img, axis=None))
    #blue
    blue_channel = src[:,:,0]
    blue_img = np.zeros(src.shape)
    blue_img[:,:,0] = blue_channel
    result.append(kurtosis(blue_img, axis=None))
    #green
    green_channel = src[:,:,1]
    green_img = np.zeros(src.shape)
    green_img[:,:,1] = green_channel
    result.append(kurtosis(green_img, axis=None))

def kurtosisHSVKey(img):
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    hsv_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2HSV)
    hue_channel = hsv_img[:, :, 0]
    saturation_channel = hsv_img[:, :, 1]
    value_channel = hsv_img[:, :, 2]
    hue_img = np.zeros(hsv_img.shape)
    saturation_img = np.zeros(hsv_img.shape)
    value_img = np.zeros(hsv_img.shape)
    hue_img[:, :, 0] = hue_channel
    saturation_img[:, :, 0] = saturation_channel
    value_img[:, :, 0] = value_channel
    result.append(kurtosis(hue_img, axis=None))
    result.append(kurtosis(saturation_img, axis=None))
    result.append(kurtosis(value_img, axis=None))

def kurtosisLabKey(img):
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    lab_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2LAB)
    lum_channel = lab_img[:, :, 0]
    a_channel = lab_img[:, :, 1]
    b_channel = lab_img[:, :, 2]
    lum_img = np.zeros(lab_img.shape)
    a_img = np.zeros(lab_img.shape)
    b_img = np.zeros(lab_img.shape)
    lum_img[:, :, 0] = lum_channel
    a_img[:, :, 0] = a_channel
    b_img[:, :, 0] = b_channel
    result.append(kurtosis(lum_img, axis=None))
    result.append(kurtosis(a_img, axis=None))
    result.append(kurtosis(b_img, axis=None))

def averageRGB(img):
    """appends average rgb channels on a picture"""
    src_img = img
    average_color_row = np.average(src_img, axis=0)
    average_color = np.average(average_color_row, axis=0)
    r=average_color[0]
    g=average_color[1]
    b=average_color[2]
    result.append(r)
    result.append(g)
    result.append(b)

def averageHSV(img):
    """appends average hsv channels on a picture"""
    src_img = img
    img_hsv = cv.cvtColor(src_img, cv.COLOR_BGR2HSV)
    average_hsv = cv.mean(img_hsv)[:3]

    h=average_hsv[0]
    s=average_hsv[1]
    v=average_hsv[2]
    result.append(h)
    result.append(s)
    result.append(v)

def averageLAB(img):
    """appends average lab channels on a picture"""
    src_img = img
    lab_img = cv.cvtColor(src_img, cv.COLOR_BGR2LAB)
    average_lab = cv.mean(lab_img)[:3]
    l=average_lab[0]
    a=average_lab[1]
    b=average_lab[2]
    result.append(l)
    result.append(a)
    result.append(b)

def GLCM(img):
    gray = color.rgb2gray(img)
    image = img_as_ubyte(gray)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max()+1
    matrix_coocurrence = skimage.feature.graycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
    contrast = skimage.feature.graycoprops(matrix_coocurrence, 'contrast')
    dissimilarity = skimage.feature.graycoprops(matrix_coocurrence, 'dissimilarity')	
    homogeneity = skimage.feature.graycoprops(matrix_coocurrence, 'homogeneity')
    energy = skimage.feature.graycoprops(matrix_coocurrence, 'energy')
    correlation = skimage.feature.graycoprops(matrix_coocurrence, 'correlation')
    result.append(correlation[0][0])
    result.append(correlation[0][1])
    result.append(correlation[0][2])
    result.append(correlation[0][3])
    result.append(contrast[0][0])
    result.append(contrast[0][1])
    result.append(contrast[0][2])
    result.append(contrast[0][3])
    result.append(dissimilarity[0][0])
    result.append(dissimilarity[0][1])
    result.append(dissimilarity[0][2])
    result.append(dissimilarity[0][3])
    result.append(homogeneity[0][0])
    result.append(homogeneity[0][1])
    result.append(homogeneity[0][2])
    result.append(homogeneity[0][3])
    result.append(energy[0][0])
    result.append(energy[0][1])
    result.append(energy[0][2])
    result.append(energy[0][3])
    
def varianceRGBKey(img):
    """appends variance of rgb channels on a picture"""
    im = Image.fromarray(img)
    stat = ImageStat.Stat(im)
    statVar = stat.var
    result.append(statVar[0])
    result.append(statVar[1])
    result.append(statVar[2])

def varianceHSVKey(img):
    """appends variance of hsv channels on a picture"""
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    mean, stats = cv.meanStdDev(img)
    result.append(stats[0]**2)
    result.append(stats[1]**2)
    result.append(stats[2]**2)

def varianceLABKey(img):
    """appends variance of lab channels on a picture"""
    img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    mean, stats = cv.meanStdDev(img)
    result.append(stats[0]**2)
    result.append(stats[1]**2)
    result.append(stats[2]**2)

def skewnessRGBKey(img):
    """appends skewness of rgb channels on a picture"""
    im = Image.fromarray(img)
    stat = ImageStat.Stat(im)
    mean = stat.mean
    median = stat.median
    stddev = stat.stddev
    result.append(3*(mean[0]-median[0])/stddev[0])
    result.append(3*(mean[1]-median[1])/stddev[1])
    result.append(3*(mean[2]-median[2])/stddev[2])

def skewnessHSVKey(img):
    """appends skewness of hsv channels on a picture"""
    if(isgray(img)):
        result.append(0)
        result.append(0)
        result.append(0)
    else:
        im = Image.fromarray(img)
        im = im.convert('HSV')
        stat = ImageStat.Stat(im)
        mean = stat.mean
        median = stat.median
        stddev = stat.stddev
        result.append(3*(mean[0]-median[0])/stddev[0])
        result.append(3*(mean[1]-median[1])/stddev[1])
        result.append(3*(mean[2]-median[2])/stddev[2])

def skewnessLABKey(img):
    """appends skewness of lab channels on a picture"""
    if(isgray(img)):
        result.append(0)
        result.append(0)
        result.append(0)
    else:
        im = Image.fromarray(img)
        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile  = ImageCms.createProfile("LAB")
        rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
        lab_im = ImageCms.applyTransform(im, rgb2lab_transform)
        stat = ImageStat.Stat(lab_im)
        mean = stat.mean
        median = stat.median
        stddev = stat.stddev
        result.append(3*(mean[0]-median[0])/stddev[0])
        result.append(3*(mean[1]-median[1])/stddev[1])
        result.append(3*(mean[2]-median[2])/stddev[2])



def load_characterictics(img):
    """returns list with all the characteristics of the picture"""
    result.clear()
    thresholdOtsuKey(img)
    circlesKey(img)  
    densityKey(img)
    meanStandardDeviationKey(img)
    kurtosisRGBKey(img)
    kurtosisHSVKey(img)
    kurtosisLabKey(img)
    averageRGB(img)
    averageHSV(img)
    averageLAB(img)
    GLCM(img)
    StandardDeviationHSV(img)
    StandardDeviationLAB(img)
    Edges(img)
    varianceRGBKey(img)
    varianceHSVKey(img)
    varianceLABKey(img)
    skewnessRGBKey(img)
    skewnessHSVKey(img)
    skewnessLABKey(img)
    return result