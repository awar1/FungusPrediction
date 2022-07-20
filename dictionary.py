import pandas as pd
from sklearn.preprocessing import StandardScaler
from characteristics import saturationStdDev,valueStdDev,aStdDev,lightnessStdDev,circles, thresholdOtsu, density, redStdDev, greenStdDev, blueStdDev, kurRed, kurBlue,kurGreen, kurHue,kurSat,kurVal,kurLab,kurA,kurB,redMean,greenMean,blueMean,avgR,avgG,avgB,avgH,avgS,avgV,avgL,avgA,avgBB,contrasttab1,contrasttab2,contrasttab3,contrasttab4,dissimilaritytab1,dissimilaritytab2,dissimilaritytab3,dissimilaritytab4,homogeneitytab1,homogeneitytab2,homogeneitytab3,homogeneitytab4,energytab1,energytab2,energytab3,energytab4,correlationtab1,correlationtab2,correlationtab3,correlationtab4,varianceRed,varianceBlue,varianceGreen,varianceH,varianceS,varianceV,varianceL,varianceA,varianceB,skewnessA,skewnessB,skewnessBlue,skewnessGreen,skewnessH,skewnessS,skewnessRed,skewnessL,skewnessV,hueStdDev,bStdDev,edgeRmean,edgeRstdDev,edgeBmean,edgeBstdDev,edgeDensity,edgeGmean,edgeGstdDev

# each micellium gets a number in range 0-3. Numbers in "for" loop are the number of pictures of that micellium that were used to create a model
fusarium_number = [0 for i in range(1,58)]
Phytophtora_number = [1 for i in range(1,21)]
trichoderma_number = [2 for i in range(1,27)]
Verticillium_number = [3 for i in range(1,26)]

def create_Dictionary():
    """Returns pandas data frames, where X is all the characterictics taken from pictures, and Y is number assigned to each micellium"""
    dictionary = {'type': fusarium_number+Phytophtora_number+trichoderma_number
                +Verticillium_number,
                'thresholdOtsu': thresholdOtsu,
                'circles': circles,
                'density': density,
                'redMean': redMean,
                'greenMean': greenMean,
                'blueMean': blueMean,
                'redStdDev': redStdDev,
                'greenStdDev':greenStdDev,
                'blueStdDev':blueStdDev,
                'kurRed':kurRed,
                'kurBlue':kurBlue,
                'kurGreen':kurGreen,
                'kurHue':kurHue,
                'kurSat':kurSat,
                'kurVal':kurVal,
                'kurLab':kurLab,
                'kurA':kurA,
                'kurB':kurB,
                'avgR':avgR,
                'avgG':avgG,
                'avgB':avgB,
                'avgH':avgH,
                'avgS':avgS,
                'avgV':avgV,
                'avgL':avgL,
                'avgA':avgA,
                'avgBB':avgBB,
                'correlationtab1':correlationtab1,
                'correlationtab2':correlationtab2,
                'correlationtab3':correlationtab3,
                'correlationtab4':correlationtab4,
                'contrasttab1':contrasttab1,
                'contrasttab2':contrasttab2,
                'contrasttab3':contrasttab3,
                'contrasttab4':contrasttab4,
                'dissimilaritytab1':dissimilaritytab1,
                'dissimilaritytab2':dissimilaritytab2,
                'dissimilaritytab3':dissimilaritytab3,
                'dissimilaritytab4':dissimilaritytab4,
                'homogeneitytab1':homogeneitytab1,
                'homogeneitytab2':homogeneitytab2,
                'homogeneitytab3':homogeneitytab3,
                'homogeneitytab4':homogeneitytab4,
                'energytab1':energytab1,
                'energytab2':energytab2,
                'energytab3':energytab3,
                'energytab4':energytab4,
                'hueStdDev':hueStdDev,
                'saturationStdDev':saturationStdDev,
                'valueStdDev':valueStdDev,
                'lightnessStdDev':lightnessStdDev,
                'aStdDev':aStdDev,
                'bStdDev':bStdDev,
                'edgeDensity':edgeDensity,
                'edgeBmean':edgeBmean,
                'edgeGmean':edgeGmean,
                'edgeRmean':edgeRmean,
                'edgeBstdDev':edgeBstdDev,
                'edgeGstdDev':edgeGstdDev,
                'edgeRstdDev':edgeRstdDev,
                'varianceRed': varianceRed,
                'varianceGreen': varianceGreen,
                'varianceBlue': varianceBlue,
                'varianceH': varianceH,
                'varianceS': varianceS,
                'varianceV': varianceV,
                'varianceL': varianceL,
                'varianceA': varianceA,
                'varianceB': varianceB,
                'skewnessRed':skewnessRed,
                'skewnessGreen':skewnessGreen,
                'skewnessBlue':skewnessBlue,
                'skewnessH':skewnessH,
                'skewnessS':skewnessS,
                'skewnessV':skewnessV,
                'skewnessL':skewnessL,
                'skewnessA':skewnessA,
                'skewnessB':skewnessB
                }
    df = pd.DataFrame(dictionary)
    scaler = StandardScaler()
    words = ['thresholdOtsu', 'circles', 'density','redMean','greenMean','blueMean','redStdDev','greenStdDev','blueStdDev',
    'kurRed','kurBlue','kurGreen', 'kurHue', 'kurSat', 'kurVal', 'kurLab', 'kurA', 'kurB',
    'avgR','avgG','avgB','avgH','avgS','avgV','avgL','avgA','avgBB',
    'correlationtab1','correlationtab2','correlationtab3','correlationtab4','contrasttab1','contrasttab2','contrasttab3','contrasttab4',
    'dissimilaritytab1','dissimilaritytab2','dissimilaritytab3','dissimilaritytab4','homogeneitytab1','homogeneitytab2','homogeneitytab3','homogeneitytab4',
    'energytab1','energytab2','energytab3','energytab4','hueStdDev','saturationStdDev','valueStdDev','lightnessStdDev','aStdDev','bStdDev','edgeDensity',
    'edgeBmean','edgeGmean','edgeRmean','edgeBstdDev','edgeGstdDev','edgeRstdDev','varianceRed','varianceGreen','varianceBlue','varianceH','varianceS','varianceV','varianceL',
    'varianceA','varianceB','skewnessRed','skewnessGreen','skewnessBlue','skewnessH','skewnessS',
    'skewnessV','skewnessL','skewnessA','skewnessB']
    
    X = df[words]
    Y = df.type

    #X = scaler.fit_transform(X)
    return X,Y