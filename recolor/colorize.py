#coding=utf-8
from sklearn.cluster import KMeans
import numpy as np
import scipy.misc
import requests
import json
import cPickle as pickle
import copy
import os 

from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from colormap import getGroupArea
from sklearn.cluster import KMeans
from skimage.future import graph
from colormap import getSparatialFea , getLabelInOrgPic , sparatialFea , getLightnessTrainSet , getFullSpatialFeaTrainSet , getContrastTrainSet2 , getbckColor , rgb2lab
from sklearn import linear_model
from skimage import io


def getBackgroundColor(im):
    return getbckColor(im)


def rgbTolab(r , g , b):
    return rgb2lab(r , g , b)
    
def getImageArray(filename):
    return scipy.misc.imread(filename , mode='RGB')

def getProductColor(im):
    temp = im.shape
    random_state = 170
    km = KMeans(n_clusters = 3 , random_state = random_state)
    im = im.reshape(-1 , 3)
    km.fit(im)
    cluster_centers = km.cluster_centers_
    return cluster_centers

# color from colormind
# curl 'http://colormind.io/api/' --data-binary '{"input":[[44,43,44],[90,83,82],"N","N","N"],"model":"default"}'
def getRecColor(color):
    url = 'http://colormind.io/api/'
    data = {}
    headers = {}
    headers['Content-Type'] = 'text/plain;charset=UTF-8'
    headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'

    data['model'] = 'default'
    data['input'] = color

    r = requests.post(url , headers = headers , data = json.dumps(data))

    if r.status_code == 200:
        res = json.loads(r.text)['result']
    else:
        res = []

    return res

def getUnaryStandScalar():
    pic_fea =  getLightnessTrainSet('./picture/sucai/')
    lightness = [item['bak_color'][0] for item in pic_feature]
    km = KMeans(n_clusters=10 , random_state=170)
    lightness = np.asarray(lightness)
    km.fit(lightness.reshape(-1 , 1))
    labels = km.labels_
    pickle.dump(km , open('km_lightness2.dat' , 'wb') , True)

    X = [[item['area_totalArea'] , item['min_seg'] , item['max_seg'] , item['segNumber']] for item in pic_feature]
    sc = StandardScaler()
    sc.fit(X)
    pickle.dump(sc , open('unary_sc.dat' , 'wb') , True)
    return sc , X , labels ,  km

def getPairwiseStandScalar():
    fullSpaFea = getFullSpatialFeaTrainSet('./picture/sucai/')
    contrast_label = getContrastTrainSet2('./font_pic/')
    contrastValues = [item[k] for item in contrast_label for k in item.keys() if k != 'label']
    km = KMeans(n_clusters = 10  , random_state = 170)
    contrastValues = np.array(contrastValues)
    km.fit(contrastValues.reshape(-1 , 1))
    pickle.dump(km , open('contrast_km.dat' , 'wb') , True)
    spaBakFontFea = copy.deepcopy(fullSpaFea)

    contrast_label_reg = []
    for item in contrast_label:
        temp = {}
        for k , v in item.items():
            if k != 'label':
                temp[os.path.basename(k)] = item[k]
            else:
                temp[k] = item[k]
        contrast_label_reg.append(temp)
    spaBakFontFeaReg = []
    for item in spaBakFontFea:
        temp = {}
        for contrastItem in contrast_label_reg:
            for k , v in contrastItem.items():
                if k!='label' and item.keys()[0] == k:
                    temp['contrast'] = v
                    a , b = contrastItem['label']
                    elem = [item[k][a] , item[k][b]]
                    temp[k] = elem
        spaBakFontFeaReg.append(temp)
    X = []
    Y = []
    for item in spaBakFontFeaReg:
        Y.append(km.predict(item['contrast']))
        keya , keyb = item.keys()
        if keya == 'contrast':
            key = keyb
        else:
            key = keya
        tmp = []
        for elem in item[key]:        
            tmp.append(elem['area_maxArea'])
            tmp.append(elem['area_totalArea'])
            tmp.append(elem['max_seg'])
            tmp.append(elem['mean_seg'])
            tmp.append(elem['min_seg'])
            tmp.append(elem['segNumber'])
            tmp.append(elem['std_seg'])
        X.append(tmp)

    sc = StandardScaler()
    sc.fit(X)
    pickle.dump(sc , open('pairwise_sc.dat' , 'wb') , True)
    return sc , X , Y , km


def getUnaryClassifier():
    sc , X , labels , _ = getUnaryStandScalar()
    X_std = sc.transform(X)
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X_std , labels)
    pickle.dump(logreg , open('logreg_classifier.dat' , 'wb') , True)
    return logreg

def getLightnessCluster():
    _ , _ , _ , km = getUnaryStandScalar()
    return km

def getPairwiseClassifier():
    sc , X , Y , _ = getPairwiseStandScalar()
    X_std = sc.transform(X)
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X_std , Y)
    pickle.dump(logreg , open('pairwise_classifier.dat' , 'wb') , True)
    return logreg

def getContrastCluster():
    _ , _ , _ , km = getPairwiseStandScalar()
    return km

def regularFea(X , filename):
    if os.path.exists(filename):
        sc = pickle.load(open(filename , 'rb'))
    else:
        sc = getUnaryStandScalar()
    #print X
    X = np.array(X)
    X = X.reshape(1  , -1)
    return sc.transform(X)


# get score about unary
def unary(X):
    X = regularFea(X , 'unary_sc.dat')
    if os.path.exists('logreg_classifier.dat'):
        logreg = pickle.load(open('logreg_classifier.dat' , 'rb'))
    else:
        logreg = getUnaryClassifier()
    #X shape should be (1 , n), n is number of features
    target = logreg.predict_proba(X)

    if os.path.exists('km_lightness2.dat'):
        km = pickle.load(open('km_lightness2.dat' , 'rb'))
    else:
        km = getLightnessCluster()
    x_bins = km.cluster_centers_
    x_plot_list = []
    for i in range(10):
        for j in range(int(target[0 , i] * 100)):
            x_plot_list.append(x_bins[i])
    data = np.array(x_plot_list)
    density = gaussian_kde(list(data.flatten()))
    xs = np.linspace(0 , 10 , 1000)
    density.covariance = lambda: .75
    density._compute_covariance()
    return density

# get score about pairwise
def pairwise(X):
    X = regularFea(X , 'pairwise_sc.dat')
    if os.path.exists('pairwise_classifier.dat'):
        logreg = pickle.load(open('pairwise_classifier.dat' , 'rb'))
    else:
        logreg = getPairwiseClassifier()

    target = logreg.predict_proba(X)
    if os.path.exists('contrast_km.dat'):
        km = pickle.load(open('contrast_km.dat' , 'rb'))
    else:
        km = getContrastCluster()
    x_bins = km.cluster_centers_
    x_plot_list = []
    for i in range(10):
        for j in range(int(target[0 , i] * 100)):
            x_plot_list.append(x_bins[i])
    data = np.array(x_plot_list)
    density = gaussian_kde(list(data.flatten()))
    xs = np.linspace(0 , 10 , 1000)
    density.covariance = lambda: .75

    density._compute_covariance()
    return density

def getColorArea(im , fea_vec):
    return getGroupArea(im , fea_vec)

# im is the piece only including text(Title layout)
def getStrength(im):
    io.imsave('font.jpg' , im)
    km = KMeans(n_clusters = 2 , random_state = 170)
    km.fit(im.reshape(-1 , 3))
    labels = km.labels_
    labels = labels.reshape(-1 , im.shape[1])
    print np.unique(labels)
    g = graph.rag_mean_color(im , labels)
    # only have 2 nodes and the strength is edges' weight
    res = g.get_edge_data(0 , 1)
    return res['weight']

def getUnaryFea(im):
    spa_fea = getSparatialFea(im)
    X = [spa_fea['area_totalArea'] , spa_fea['min_seg'] , spa_fea['max_seg'] , spa_fea['segNumber']]
    return np.array(X)

def getPairwiseFea(im , font_color , bak_color):
    font_vec , bak_vec = getLabelInOrgPic(im , font_color , bak_color)
    res = [sparatialFea(im , font_vec) , sparatialFea(im , bak_vec)]
    X = []
    for elem in res:
        X.append(elem['area_maxArea'])
        X.append(elem['area_totalArea'])
        X.append(elem['max_seg'])
        X.append(elem['mean_seg'])
        X.append(elem['min_seg'])
        X.append(elem['segNumber'])
        X.append(elem['std_seg'])
    return X


