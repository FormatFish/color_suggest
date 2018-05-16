#coding=utf-8
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from PIL import Image ,ImageDraw,ImageOps
import scipy.misc
from sklearn.cluster import KMeans
import networkx as nx
from skimage import data, io, segmentation, color ,morphology , measure
from skimage.future import graph
from matplotlib import pyplot as plt

import requests
import json
import base64
import os
import glob
import re


def img2np(filename):
    #print filename
    img = Image.open(filename)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    im = np.asarray(img)
    return im


def mean_shift(im):
    tmp = im.shape
    im = im.reshape((-1 , 3))
    bandwidth = estimate_bandwidth(im, quantile=0.1, n_samples=1500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(im)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print "number of estimated cluster :%d" % n_clusters_

    imNew = np.zeros(im.shape)
    l = ms.predict(im)

    area = np.zeros((n_clusters_ , 1))
    cnt = 0
    for i in range(len(l)):
        imNew[i] = cluster_centers[l[i]]
        area[l[i]] += 1


    imNew = imNew.reshape(tmp)
    area = area * 1.0 / area.sum() * 100
    #node_labels = zip(cluster_centers , area)
    scipy.misc.imsave('outfile.jpg' , imNew)

    labels = labels.reshape((-1 , tmp[1]))
    return labels , area , cluster_centers , ms , imNew

def display(g , title , node_labels , color , rag_filename):
    pos = nx.circular_layout(g)
    plt.figure(figsize=(18 , 17),dpi=80)
    plt.title(title)
    #plt.text(0 , 0 , u'nodes: extracted color from image' , fontsize=20)
    #plt.text(0 , 30 , u'edges: color connect' , fontsize=20)
    #plt.text(0 , 60 , u'weight: the Euclidean distance' , fontsize=20)
    nx.draw(g , pos)
    nx.draw_networkx_edge_labels(g , pos , font_size = 10)
    #print node_labels
    nodesLabels = {k:str(round(v , 3)) + '%' for k , v in enumerate(node_labels)}
    #color = [(item[0]/255 , item[1]/255 , item[2]/255) for item in color]
    color = [(item[0] , item[1] , item[2]) for item in color]
    #colorMap = {k:v for k , v in enumerate(color)}
    #print color
    #print nodesLabels
    #nodes_size = [item * 2000 for item in node_labels]
    nx.draw_networkx_labels(g , pos , nodesLabels , font_size = 10)
    nx.draw_networkx_nodes(g , pos , node_color = color , node_size = 800)
    plt.savefig('./'+rag_filename)
    #plt.show()


def RAG(im , rag_filename='colorNet.png'):
    #labels , node_labels , cluster_centers , _= mean_shift(im);
    labels , node_labels , cluster_centers , _= k_means(im);
    #print node_labels
    g = graph.rag_mean_color(im , labels , mode='distance')
    display(g , 'Region Adjacency Graph' , node_labels , cluster_centers , rag_filename)


def getAccessToken():
    clientId = "doax2vFEhtvQXZFLoIRsSHBG"
    clientSecret = "GsHeaft6npMoBY3crSlTQFfCn8uF4Eex"
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id="+ clientId + "&client_secret=" + clientSecret
    access = requests.post(url)

    token = json.loads(access.text)

    return token['access_token']


def getTextInfo(filename , save_name = 'temp.jpg'):
    headers = {'Content-Type':'application/x-www-form-urlencoded'}
    baseUrl = u"https://aip.baidubce.com/rest/2.0/ocr/v1/general?access_token=" + getAccessToken()
    data = {'image': base64.b64encode(open(filename , 'rb').read())}
    data['recognize_granularity'] = 'small'
    data['detect_direction'] = True
    data['vertexes_location'] = True

    print '当前处理的文件名:' , filename

    r = requests.post(baseUrl , data = data , headers = headers)
    print r
    if r.status_code != 200:
        print 'Sorry about network: text ocr server is not connected'
        return None

    try:
        info = json.loads(r.text)
    except:
        print 'There is no json obj'
        return None
    else:        
        #print info['words_result_num']
        #print info
        if info['words_result_num'] == 0:
            return None

        wordsRes = info['words_result']
        words = wordsRes[0]['words']
        location = wordsRes[0]['location'] # 获取整个字符串的box

        captha = Image.open(filename)
        if captha.mode != 'RGB':
            captha = captha.convert('RGB')
        draw = ImageDraw.ImageDraw(captha)
        x = location['left']
        y = location['top']
        w = location['width']
        h = location['height']
        #draw.rectangle((x , y , x + w , y + h) , outline = 'red')

        chars = wordsRes[0]['chars']
        charMap = {}
        for item in chars:
            charMap[item['char']] = item['location']

        imgChar = [];
        for item in charMap.values():
                x = item['left']
                y = item['top']
                w = item['width']
                h = item['height']
                imgChar.append(captha.crop((x , y , x + w , y + h)))
                draw.rectangle((x , y , x + w , y + h) , outline = 'blue')
        
        captha.save(save_name)
        for i in range(len(imgChar)):
        	imgChar[i].save(str(i) + '.jpg')

        imgText = captha.crop((x , y , x + w , y+h))

        return imgChar

    '''
    chars = wordsRes[0]['chars']
    charMap = {}
    for item in chars:
        charMap[item['char']] = item['location']
    for item in charMap.values():
        x = item['left']
        y = item['top']
        w = item['width']
        h = item['height']
        draw.rectangle((x , y , x + w , y + h) , outline = 'blue')
    '''
    #return info



def recogTextColor(cluster , img):
    #img.save('text.jpg')
    labels , area , textCenter , _ , _= k_means(np.asarray(img))
    #RAG(np.asarray(img))
    #print textCenter
    #print area
    dominat_color = textCenter[np.argmax(area)]

    l = cluster.predict([dominat_color,])#所谓的主色调在原图色板中的位置
    return l



def k_means(im):
    tmp = im.shape
    #print tmp
    #print im
    im = color.rgb2lab(im)
    #im = color.rgb2hsv(im)

    im = im.reshape((-1 , 3))
    #bandwidth = estimate_bandwidth(im, quantile=0.1, n_samples=1500)
    #ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    random_state = 170
    km = KMeans(n_clusters=5, random_state=random_state)
    #labels = km.fit_predict(im)
    km.fit(im)
    labels = km.labels_
    cluster_centers = km.cluster_centers_
    #print 'asdasdas\n',cluster_centers
    #labels_unique = np.unique(labels)
    n_clusters_ = len(cluster_centers)
    #print "number of estimated cluster :%d" % n_clusters_

    imNew = np.zeros(im.shape)
    #l = ms.predict(im)

    #area = np.zeros((n_clusters_ , 2))
    #cnt = 0
    #cluster_centers = cluster_centers.astype(np.int64)
    area = np.zeros((n_clusters_ , 1))  
    for i in range(len(labels)):
        imNew[i] = cluster_centers[labels[i]]
    	area[labels[i]] += 1


    imNew = imNew.reshape(tmp)      

    imNew = color.lab2rgb(imNew)
    u = imNew.reshape((-1 , 3))
    b = np.ascontiguousarray(u).view(np.dtype((np.void, u.dtype.itemsize * u.shape[1])))
    _, idx = np.unique(b, return_index=True)

    cluster_centers = u[idx]
    #imNew = color.hsv2rgb(imNew)
    area = area * 1.0 / area.sum() * 100
    #print cluster_centers
    #print area
    #print area * 1.0 / area.sum()
    scipy.misc.imsave('outfile.jpg' , imNew)
    labels = labels.reshape((-1 , tmp[1]))
    return labels , area , cluster_centers , km , imNew

def getbckColor(im):
    print im.shape
    random_state = 170
    km = KMeans(n_clusters = 5 , random_state = random_state)
    km.fit(im.reshape(-1 , 3))
    labels = km.labels_
    cluster_centers = km.cluster_centers_

    area = [(labels == item).sum() for item in range(5)]
    area = np.asarray(area)
    #print area
    l = km.predict(cluster_centers)
    index = np.argwhere(l == np.argmax(area))
    #print cluster_centers[index]
    #print np.argmax(area)
    '''
    cluster_centers = cluster_centers.astype(np.int64)
    n_clusters_ = len(cluster_centers)
    imNew = np.zeros(im.shape)

    area = np.zeros((n_clusters_ , 1))  
    for i in range(len(labels)):
        imNew[i] = cluster_centers[labels[i]]
        area[labels[i]] += 1
    '''
    print cluster_centers
    print index
    #print area
    return cluster_centers[int(index)] , np.argmax(area) , labels.reshape(-1 , im.shape[1])



def gamma(x):
    if x > 0.04045:
        return ((x + 0.055) * 1.0 / 1.055) ** 2.4
    else:
        return x * 1.0 / 12.92

def xyz2lab(t):
    if t > (6 * 1.0 / 29) ** 3:
        return t ** (1.0 /3)
    else:
        return 1.0 / 3 * ((29.0 / 6) ** 2) * t + 4.0 / 29


def rgb2lab(r , g , b):
    R = gamma(r * 1.0 / 255)
    G = gamma(g * 1.0 / 255)
    B = gamma(b * 1.0 / 255)
 
    M = np.array([[0.4124,0.3576,0.1805] , 
                [0.2126,0.7152,0.0722] , 
                [0.0193,0.1192,0.9505]])
    RGB = np.array([[R] , 
                    [G] , 
                    [B]])

    XYZ = np.dot(M , RGB)

    x , y , z = [float(item) for item in XYZ]

    l = 116 * xyz2lab(y * 1.0 / 100.0) - 16
    a = 500 * (xyz2lab(x * 1.0/ 95.047) - xyz2lab(y * 1.0 / 100.0))
    b = 200 * (xyz2lab(y * 1.0/ 100.0) - xyz2lab(z * 1.0 / 108.88))

    return l , a , b

def edu_distance(a , b):
    '''
    @param a: numpy
    @param b: numpy
    '''
    return (((a - b)**2).sum())**0.5


#get gray img
# Attention pic's format must be rgb
'''
def getOneGroup(im , fea_vec):
    
    @im numpy type == numpy.int64
    @return 2D array of picture(gray pic)
    
    temp = im.shape
    im = im.reshape(-1 , 3)
    idx = np.where(np.all(im == fea_vec , axis=1))
    im[idx] = np.asarray([0.,0.,0.,])

    im = im.reshape(temp)
    return color.rgb2gray(im)
'''
#group area / total area

def relativeSize(im , fea_vec , max_area):
    '''
    max_area = getMaxGroupArea()
    if using this func, The first of all is runing getMaxGroupArea
    '''
    totalArea = im.shape[0] * im.shape[1]
    area = getGroupArea(im , fea_vec) 

    return area * 1.0 / totalArea , area * 1.0 / max_area

def relativeSize2(im , fea_vec):
    '''
    max_area = getMaxGroupArea()
    if using this func, The first of all is runing getMaxGroupArea
    '''
    totalArea = im.shape[0] * im.shape[1]
    area = getGroupArea(im , fea_vec) 

    return area * 1.0 / totalArea

def getSegment(im , fea_vec):
    '''
    temp = im.shape
    im = im.reshape(-1 , 3)
    idx = np.where(np.all(im == fea_vec , axis=1))
    im[idx] = np.asarray([0.,0.,0.,])

    im = im.reshape(temp)
    imNew = color.rgb2gray(im)
    imNew = (imNew > 0) * 1
    imNew = 1 - imNew
    '''
    imNew = (im == fea_vec) * 1
    # chull = morphology.convex_hull_object(imNew)
    # dst=morphology.remove_small_objects(chull*1,min_size=300,connectivity=2)
    # labels=measure.label(dst,connectivity=2)
    region = measure.regionprops(imNew)
    return region



def getGroupArea(im , fea_vec):    
    region = getSegment(im , fea_vec)
    area = 0
    for item in region:
        area += item.area
    return area

# get max area
def getMaxGroupArea(im , fea_vecs):
    max_area = 0
    for item in fea_vecs:
        area = getGroupArea(im , item)
        if area > max_area:
            max_area = area


    return max_area

#Number of Segments
def getRatioOfSeg(im , fea_vec , nums):
    region = getSegment(im , fea_vec)
    return len(region)* 1.0 / nums



def getTotalSegments(im , fea_vecs):
    nums = 0;
    for item in fea_vecs:
        region = getSegment(im , item)
        nums += len(region)

    return nums



def getCovMatrix(im , fea_vec):
    region = getSegment(im , fea_vec)
    matrix = []
    for item in region:
        matrix.append(item.centroid)
    return np.cov(matrix)


def getStaticsInfo(im , fea_vec):
    region = getSegment(im , fea_vec)
    sizes = []
    for item in region:
        sizes.append(item.area)

    sizes = np.asarray(sizes)
    return np.max(sizes)* 1.0/ np.sum(sizes) , np.min(sizes)* 1.0/ np.sum(sizes) , np.mean(sizes) * 1.0 / np.sum(sizes) , np.std(sizes)

def saturation(color):
    return (color[1] ** 2 + color[2] ** 2) ** 0.5 / (((color**2).sum())** 0.5)

def getLightness(im):
    bak_color , _ , _  = getbckColor(im)
    bak_color = np.asarray(rgb2lab(bak_color[0] , bak_color[1] , bak_color[2]))
    return bak_color

'''
def unary():
    color  = getLightnessTrainSet('./picture/sucai/')
    lightness = [item[0] for item in color]
    random_state = 170
    km = KMeans(n_clusters=10, random_state=random_state)
    labels = km.labels_
    cluster_centers = km.cluster_centers_
'''
def getSparatialFea(im):
    bak_color , bak_color_label , imNew = getbckColor(im)
    #print fea_vecs
    #print imNew
    fea_vecs = range(5)
    fea_vec = bak_color_label
    bak_color = np.asarray(rgb2lab(bak_color[0] , bak_color[1] , bak_color[2]))
    print 'bak_color' , bak_color
    #max_area = getMaxGroupArea(imNew , fea_vecs)
    #print 'max_area' , max_area
    #area_totalArea , area_maxArea = relativeSize(imNew , fea_vec , max_area)
    area_totalArea = relativeSize2(imNew , fea_vec)
    print 'area_totalArea' , area_totalArea
    #segSpread = getCovMatrix(imNew , fea_vec)
    #print 'segSpread', segSpread
    max_seg , min_seg , mean_seg , std_seg = getStaticsInfo(imNew , fea_vec)
    print 'max_seg' , max_seg
    nums = getTotalSegments(imNew , fea_vecs)
    segNumber = getRatioOfSeg(imNew , fea_vec , nums)
    print 'segNumber', segNumber
    spa_feature = {}
    spa_feature['bak_color'] = bak_color
    spa_feature['area_totalArea'] = area_totalArea
    #spa_feature['area_maxArea'] = area_maxArea
    #spa_feature['segSpread'] = segSpread
    spa_feature['max_seg'] = max_seg
    spa_feature['min_seg'] = min_seg
    spa_feature['mean_seg'] = mean_seg
    spa_feature['std_seg'] = std_seg
    spa_feature['segNumber'] = segNumber
    return spa_feature

def sparatialFea(im , fea_vec):
    km = KMeans(n_clusters = 5 , random_state = 170)
    km.fit(im.reshape(-1 , 3))
    label_image = km.labels_
    label_image = label_image.reshape(-1 , im.shape[1])


    fea_vecs = range(5)
    #bak_color = np.asarray(rgb2lab(bak_color[0] , bak_color[1] , bak_color[2]))
    #print 'bak_color' , bak_color
    max_area = getMaxGroupArea(label_image , fea_vecs)
    #print 'max_area' , max_area
    area_totalArea , area_maxArea = relativeSize(label_image , fea_vec , max_area)
    #area_totalArea = relativeSize2(imNew , fea_vec)
    print 'area_totalArea , area_maxArea' , area_totalArea, area_maxArea
    #segSpread = getCovMatrix(imNew , fea_vec)
    #print 'segSpread', segSpread
    max_seg , min_seg , mean_seg , std_seg = getStaticsInfo(label_image , fea_vec)
    print 'max_seg' , max_seg
    nums = getTotalSegments(label_image , fea_vecs)
    segNumber = getRatioOfSeg(label_image , fea_vec , nums)
    print 'segNumber', segNumber
    spa_feature = {}
    #spa_feature['bak_color'] = bak_color
    spa_feature['area_totalArea'] = area_totalArea
    spa_feature['area_maxArea'] = area_maxArea
    #spa_feature['segSpread'] = segSpread
    spa_feature['max_seg'] = max_seg
    spa_feature['min_seg'] = min_seg
    spa_feature['mean_seg'] = mean_seg
    spa_feature['std_seg'] = std_seg
    spa_feature['segNumber'] = segNumber
    return spa_feature

def getTextSparatialFea(filename):
    #feature_list = []
    im , area , cluster_centers , cluster , imNew = k_means(img2np(filename))
    #RAG(img2np(filename) , rag_filename = 'main_rag.jpg')
    imgChar = getTextInfo(filename)
    if imgChar is None:
        return None
    mp = {}
    for i in range(len(imgChar)):
        tmp = recogTextColor(cluster , imgChar[i])[0]
        #print tmp
        if(mp.has_key(tmp)):
            mp[tmp] += 1
        else:
            mp[tmp] = 1
    max_v = 0
    max_k = 0
    for k , v in mp.items():
        if v > max_v:
            max_v = v
            max_k = k

    fea_vecs = range(5)
    fea_vec = max_k
    bak_color = np.asarray(rgb2lab(bak_color[0] , bak_color[1] , bak_color[2]))
    print 'bak_color' , bak_color
    max_area = getMaxGroupArea(im , fea_vecs)
    print 'max_area' , max_area
    area_totalArea , area_maxArea = relativeSize(im , fea_vec , max_area)
    print 'area_totalArea , area_maxArea' , area_totalArea, area_maxArea
    #segSpread = getCovMatrix(imNew , fea_vec)
    #print 'segSpread', segSpread
    max_seg , min_seg , mean_seg , std_seg = getStaticsInfo(im , fea_vec)
    print 'max_seg' , max_seg
    nums = getTotalSegments(im , fea_vecs)
    segNumber = getRatioOfSeg(im , fea_vec , nums)
    print 'segNumber', segNumber
    spa_feature = {}
    spa_feature['bak_color'] = bak_color
    spa_feature['area_totalArea'] = area_totalArea
    spa_feature['area_maxArea'] = area_maxArea
    #spa_feature['segSpread'] = segSpread
    spa_feature['max_seg'] = max_seg
    spa_feature['min_seg'] = min_seg
    spa_feature['mean_seg'] = mean_seg
    spa_feature['std_seg'] = std_seg
    spa_feature['segNumber'] = segNumber

    return spa_feature


def features(filename):
    #filename = 'input.jpg'
    feature_list = []
    _ , area , cluster_centers , cluster , imNew = k_means(img2np(filename))
    #RAG(img2np(filename) , rag_filename = 'main_rag.jpg')
    imgChar = getTextInfo(filename)
    if imgChar is None:
        return None
    mp = {}
    for i in range(len(imgChar)):
        tmp = recogTextColor(cluster , imgChar[i])[0]
        #print tmp
        if(mp.has_key(tmp)):
            mp[tmp] += 1
        else:
            mp[tmp] = 1
    max_v = 0
    max_k = 0
    for k , v in mp.items():
        if v > max_v:
            max_v = v
            max_k = k
    #max_k should be text color label which in whole five palette
    text_color = cluster_centers[max_k]
    text_color *= 255
    text_color = np.asarray(rgb2lab(text_color[0] , text_color[1],text_color[2])) 
    bak_color , fea_vecs , imNew = getbckColor(img2np(filename))
    fea_vec = bak_color
    bak_color = np.asarray(rgb2lab(bak_color[0] , bak_color[1] , bak_color[2]))
    lightness = bak_color[0]
    perceptual_diff = edu_distance(text_color , bak_color)
    relativa_L = abs(text_color[0] - bak_color[0])
    relative_S = abs(saturation(text_color) - saturation(bak_color))
    segame_l = edu_distance(text_color[0] , bak_color[0])
    segame_a = edu_distance(text_color[1] , bak_color[1])
    segame_b = edu_distance(text_color[2] , bak_color[2])
    chromatic_diff = (segame_a ** 2 + segame_b ** 2) / (segame_a ** 2 + segame_b ** 2 + segame_l ** 2)


    #im's type must be np.int64, and must be dealed by KMeans/Meanshift
    #fea_vecs is the unique color group , such as cluster_center
    #fea_vec is the color which want analysis
    #imNew = imNew.astype(np.int64)
    #b = np.ascontiguousarray(imNew).view(np.dtype((np.void, imNew.dtype.itemsize * imNew.shape[1])))
    #_, idx = np.unique(b, return_index=True)
    #fea_vecs = imNew[idx]

    #fea_vec is special color, Now I use backgroud color

    max_area = getMaxGroupArea(imNew , fea_vecs)
    area_totalArea , area_maxArea = relativeSize(imNew , fea_vec , max_area)
    segSpread = getCovMatrix(imNew , fea_vec)
    staticInfo = getStaticsInfo(imNew , fea_vec)
    segNumber = getRatioOfSeg(imNew , fea_vec , nums)



    feature_list.append(perceptual_diff)
    feature_list.append(relativa_L)
    feature_list.append(relative_S)
    feature_list.append(chromatic_diff)
    #print 'the text color is :', cluster_centers[max_k]
    #print 'the background color is :' , getbckColor(img2np(filename))
    #print cluster_centers
    #print area
    #RAG(img2np(filename))
    #print recogTextColor(cluster , cluster_centers , getTextInfo(filename))
    #getTextInfo(filename)
    #print recogTextColor(cluster , cluster_centers , np.asarray(Image.open('text1.jpg')))
    #RAG(img2np(filename))
    return feature_list

import cPickle as pickle

def getTrainSet(path):
    pic_list = os.listdir(path)
    pic_feature = []
    if os.path.exists('features.dat'):
        pic_feature = pickle.load(open('features.dat' , 'rb'))
    else:
        for item in pic_list:
            tmp = features(path + item)
            if tmp is None:
                continue
            pic_feature.append(tmp)
        pickle.dump(pic_feature , open('features.dat' , 'wb') , True)

    print pic_feature
    return pic_feature

def getLightnessTrainSet(path):
    pic_list = os.listdir(path)
    pic_feature = []
    if os.path.exists('sparatialFea.dat'):
        pic_feature = pickle.load(open('sparatialFea.dat' , 'rb'))
    else:
        for item in pic_list:
            tmp = getSparatialFea(img2np(path + item))
            print item
            print tmp
            if tmp is None:
                continue
            pic_feature.append(tmp)
        pickle.dump(pic_feature , open('sparatialFea.dat' , 'wb') , True)

    print pic_feature
    return pic_feature

def getContrastTrainSet(path):
    pic_list = os.listdir(path)
    pic_feature = []
    if os.path.exists('textSparatialFea.dat'):
        pic_feature = pickle.load(open('textSparatialFea.dat' , 'rb'))
    else:
        for item in pic_list:
            tmp = getTextSparatialFea(img2np(path + item))
            print item
            print tmp
            if tmp is None:
                continue
            pic_feature.append(tmp)
        pickle.dump(pic_feature , open('textSparatialFea.dat' , 'wb') , True)

    print pic_feature
    return pic_feature



def textGrayPic(path):
    cnt = 0
    pic_list = os.listdir(path)
    save_path = './text/'

    for item in pic_list:
        if cnt == 10:
            break
        im , area , cluster_centers , cluster , imNew = k_means(img2np(path + item))
        #RAG(img2np(filename) , rag_filename = 'main_rag.jpg')
        imgChar = getTextInfo(path + item)
        if imgChar is None:
            return None
        mp = {}
        for i in range(len(imgChar)):
            tmp = recogTextColor(cluster , imgChar[i])[0]
            #print tmp
            if(mp.has_key(tmp)):
                mp[tmp] += 1
            else:
                mp[tmp] = 1
        max_v = 0
        max_k = 0
        for k , v in mp.items():
            if v > max_v:
                max_v = v
                max_k = k

        fea_vecs = range(5)
        fea_vec = max_k

        text_gray = (im == fea_vec) *1
        scipy.misc.imsave(save_path+str(cnt)+'.jpg' , text_gray)
        cnt += 1


def getGrayTrainSet(path):
    #filename='input.jpg'
    pic_list = os.listdir(path)
    km = pickle.load(open('km_lightness.dat' , 'rb'))
    #pic_list=["test.jpg"]
    save_path = './train/'

    cnt=0
    for item in pic_list:
        print item
        bak_color , bak_color_label , imNew = getbckColor(img2np(path + item))

        bak_color = np.asarray(rgb2lab(bak_color[0] , bak_color[1] , bak_color[2]))
        label = km.predict(bak_color[0])
        im = (imNew == bak_color_label) * 1
        #thumb = ImageOps.fit(im, (128,128), Image.ANTIALIAS)
        #scipy.misc.imsave(save_path+str(cnt)+'.jpg' , im)
        if os.path.exists(save_path + str(label)) == False:
            os.mkdir(save_path + str(label))
        #scipy.misc.imsave('ooooo.jpg' , im)
        #if os.path.isfile(save_path+str(label)+'/'+str(cnt)+'.jpg') == False:
        scipy.misc.imsave(save_path+str(label)+'/'+str(cnt)+'.jpg' , im)
        cnt += 1


def getLabelInOrgPic(im , colora , colorb):
    #path = os.path.join('./picture/sucai' , filename)
    km = KMeans(n_clusters = 5  , random_state = 170)
    #im = img2np(path)
    km.fit(im.reshape(-1 , 3))

    return [int(km.predict(colora.reshape(1 , -1))) , int(km.predict(colorb.reshape(1 , -1)))]



# get contrast but need use id( the id just is filename)
def getContrastTrainSet2(path):
    contrast_label = []
    if os.path.exists('contrast_label.dat'):
        contrast_label = pickle.load(open('contrast_label.dat' , 'rb'))
    else:
        pic_path = os.path.join(path , '*g')
        #print pic_path
        pic_list = glob.glob(pic_path)
        pattern = re.compile('.+?\.jpg')
        #print pic_list
        font_pic = {}
        for item in pic_list:
            key = pattern.match(item).group()
            if font_pic.has_key(key):
                font_pic[key].append(io.imread(item))
            else:
                font_pic[key] = [io.imread(item)]

        #pic = [os.path.basename(item) for item in font_pic.keys()]
        #origin_pic = [orign_path + item for item in pic if os.path.isfile(orign_path + item)]
        random_state = 170
        
        pic = []
        
        for k , v in font_pic.items():
            contrast = {}
            pic_contrast = []
            km = KMeans(n_clusters = 2 , random_state = random_state)
            #cluster = KMeans(n_clusters = 5 , random_state = 170)
            print k
            #last_colora = 0
            #last_colorb = 0
            for item in v:
                item = item.reshape(-1 , 3)
                km.fit(item)
                color_a , color_b = km.cluster_centers_
                #last_colora , last_colorb = colora , colorb
                color_lab_a = np.array(rgb2lab(color_a[0] , color_a[1] , color_a[2]))
                color_lab_b = np.array(rgb2lab(color_b[0] , color_b[1] , color_b[2]))
                rela_l = abs(color_lab_a[0] - color_lab_b[0])
                pic_contrast.append(rela_l)
            # use the last piece for spa feature
            contrast['label'] = getLabelInOrgPic(img2np(os.path.join('./picture/sucai' , os.path.basename(k))) , color_a , color_b)
            contrast[k] = np.mean(np.array(pic_contrast)) 
            print contrast
            contrast_label.append(contrast)
        pickle.dump(contrast_label , open('contrast_label.dat' , 'wb') , True)

    #print contrast

    return contrast_label

def getFullSpatialFeaTrainSet(path):
    random_state = 170
    km = KMeans(n_clusters = 5 , random_state = random_state)
    contrast = getContrastTrainSet2('./font_pic/')
    pic_name = [os.path.basename(item) for item in contrast.keys()]
    features = []

    if os.path.exists('fullSpatialFea.dat'):
        features = pickle.load(open('fullSpatialFea.dat' , 'rb'))
    else:
        for item in pic_name:            
            if os.path.isfile(path + item): 
                print item           
                img = scipy.misc.imread(path + item , mode="RGB")
                tmp = img.shape
                img = img.reshape(-1 , 3)
                km.fit(img)
                labels = km.labels_
                imNew = labels.reshape(-1 , tmp[1])
                fea_vecs = range(5)
                pic_feature = {}
                pic_info = []
                for fea_vec in fea_vecs:
                    
                    #fea_vec = bak_color_label
                    #bak_color = np.asarray(rgb2lab(bak_color[0] , bak_color[1] , bak_color[2]))
                    #print 'bak_color' , bak_color
                    max_area = getMaxGroupArea(imNew , fea_vecs)
                    print 'max_area' , max_area
                    area_totalArea , area_maxArea = relativeSize(imNew , fea_vec , max_area)
                    print 'area_totalArea , area_maxArea' , area_totalArea, area_maxArea
                    #segSpread = getCovMatrix(imNew , fea_vec)
                    #print 'segSpread', segSpread
                    max_seg , min_seg , mean_seg , std_seg = getStaticsInfo(imNew , fea_vec)
                    print 'max_seg' , max_seg
                    nums = getTotalSegments(imNew , fea_vecs)
                    segNumber = getRatioOfSeg(imNew , fea_vec , nums)
                    print 'segNumber', segNumber            
                    #spa_feature['bak_color'] = bak_color
                    spa_feature = {}
                    #spa_feature['filename'] = item
                    spa_feature['area_totalArea'] = area_totalArea
                    spa_feature['area_maxArea'] = area_maxArea
                    #spa_feature['segSpread'] = segSpread
                    spa_feature['max_seg'] = max_seg
                    spa_feature['min_seg'] = min_seg
                    spa_feature['mean_seg'] = mean_seg
                    spa_feature['std_seg'] = std_seg
                    spa_feature['segNumber'] = segNumber
                    pic_info.append(spa_feature)
                pic_feature[item] = pic_info
            else:
                print item ,' is not found'
                continue
            features.append(pic_feature)
        pickle.dump(features , open('fullSpatialFea.dat' , 'wb') , True)

    print features
    return features





if __name__ == '__main__':
    #filename="input.jpg"
    #bak_color , bak_color_label , imNew = getbckColor(img2np(filename))
    #print bak_color
    #fea_vecs = range(5)
    #bak_color = np.asarray(rgb2lab(bak_color[0] , bak_color[1] , bak_color[2]))
    #print 'bak_color' , bak_color
    #max_area = getMaxGroupArea(imNew , fea_vecs)
    #print 'max_area' , max_area
    #_ , area , cluster_centers , cluster = k_means(img2np(filename))
    #RAG(img2np(filename) , rag_filename = 'main_rag.jpg')
    #features('/Users/inlab/Downloads/testpp.png')
    #getTrainSet('./picture/sucai/')
    #getLightnessTrainSet('./picture/sucai/')
    #getTextInfo('/Users/inlab/Downloads/test-segment.jpg')
    #k_means(img2np)
    #getGrayTrainSet('./picture/sucai/')
    #textGrayPic('./picture/sucai/')
    #filename='texttest.jpg'
    #getTextInfo(filename , save_name = 'hhhh.jpg')
    #getContrastTrainSet('./picture/sucai/')
    getContrastTrainSet2('./font_pic/')
    #getFullSpatialFeaTrainSet('./picture/sucai/')









