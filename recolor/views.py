#coding=utf-8
from django.shortcuts import render
from django.shortcuts import render_to_response
from enum import Enum
import layout
import palettes
import option
import json
from PIL import Image
import numpy as np
from skimage import io , color , img_as_ubyte
from sklearn.cluster import KMeans
import scipy.misc
import requests
import os
from io import BytesIO
from itertools import permutations
import colormap

# Json 返回模板的参数(字典)
def getTemplateInfo(filename = 'templateM.json' , templateId = '1'):
    f = file(filename)
    s = json.load(f)
    #print templateId
    #print s.keys()
    return s[templateId]


# 筛选模板,返回模板id
def templateSelect(size , Images):
    ratioCanvas = size[0] * 1.0 / size[1]
    image = Image.open(Images[0])
    imageRatio = image.size[0] * 1.0 / image.size[1]
    templateId = '1'
    if ratioCanvas > 0.67 and ratioCanvas < 2:
        if imageRatio >0.67 and imageRatio < 2:
            templateId = '2'
        elif imageRatio < 0.67:
            templateId = '3'
        elif imageRatio > 2:
            templateId = '1'
    elif ratioCanvas > 2:
        templateId = '4'
    elif ratioCanvas < 0.67:
        templateId = '6'

    #templateId = '1'
    return templateId


# area 存放的是框（应该是要元素适应的，这里只是获取）
# pos 是起始位置
def getElementInfo(Images , size , template):
    templateId = templateSelect(size , Images)
    templateInfo = getTemplateInfo(template , templateId)
    data = templateInfo['data']['children']
    pos = {}
    area = {}

    children = data

    for key , value in children.items():
        area[key] = (value['w'] * size[0] , value['h'] * size[1])
        pos[key] = (int(value['x'] * size[0]) , int(value['y'] * size[1]))
    return pos , area


def getProductColor(im , mask):
    idx = (mask == 255)
    temp = im.shape
    random_state = 170
    km = KMeans(n_clusters = 2 , random_state = random_state)
    im_mask = im[idx]
    # im = im.reshape(-1 , 3)
    km.fit(im_mask)
    cluster_centers = km.cluster_centers_
    return [list(item) for item in list(cluster_centers.astype(np.int))] + ["N" , "N" , "N"]


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

def modifyBakColor(background , palette):
    background_image = background.background_image
    background_type = background.background_type
    #bak = io.imread(background_image)
    bak = scipy.misc.imread(background_image  , mode='RGB')
    rows , cols , channel = bak.shape
    if background_type == layout.Background_type.single:        
        for row in range(rows):
            for col in range(cols):
                bak[row , col] = [palette[-3][0] , palette[-3][1] , palette[-3][2]]
        scipy.misc.imsave(background_image , bak)

    elif background_type == layout.Background_type.dual:
        km = KMeans(n_clusters = 2 , random_state = 170)
        km.fit(bak.reshape(-1 , 3))
        labels = km.labels_
        bak_color = [palette[-3][0]*1.0/255 , palette[-3][1]*1.0/255 , palette[-3][2]*1.0/255]
        sub_bak_color = [palette[-2][0]*1.0/255 , palette[-2][1]*1.0/255 , palette[-2][2]*1.0/255]
        dst = color.label2rgb(labels.reshape(-1 , bak.shape[1]) , colors = [bak_color , sub_bak_color])
        scipy.misc.imsave(background_image , dst)
    else:
        pass
    return background
# Create your views here.
font_type = {'1':'msyh.ttf'}

def index(request):
    return render_to_response('index.html' , {'flag':False})
'''
def recolor(request):
    if request.method == 'POST':
        title_text = request.POST['title']
        title_font = font_type[request.POST['title_font']]

        subtitle_text = request.POST['subtitle']
        print subtitle_text
        subtitle_font = font_type[request.POST['subtitle_font']]

        
        # img = request.FILES['image']
        # with open('image.png' , 'wb+') as des:
        #     des.write(img.read())
        
        back = request.FILES['background']
        with open('background.png' , 'wb+') as desBack:
            desBack.write(back.read())
        
        case = int(request.POST['case'])
        if case == 1:
            images = ['./static/img/case_image_1.png']
        else:
            images = ['./static/img/case_image_2.png']


        background_image = 'background.png'
        background_type = layout.Background_type(int(request.POST['background_type']))

        #palette = getRecColor(getProductColor(scipy.misc.imread(images[0] , mode='RGB')))
        pal = palettes.Palette(getProductColor(scipy.misc.imread(images[0] , mode='RGB')))
        palette = pal.getRecColorFromColormind()
        #palette = [[49,47,49],[91,83,81],[133,155,143],[226,209,167],[235,198,126]]
        palette = [(item[0] , item[1] , item[2]) for item in palette]
        
        title_color = palette[-1]
        subtitle_color = palette[-1]
        size = (int(request.POST['w']) , int(request.POST['h']))

        pos , area = getElementInfo(images , size , template = 'templateM.json')
        title = layout.Text(title_text , title_font , title_color)
        subtitle = layout.Text(subtitle_text , subtitle_font , subtitle_color)
        background = layout.Background(background_image , background_type)

        color_change = option.Option(background , palette)
        #background = modifyBakColor(background , palette)
        curLayout = layout.Layout(pos , area , size , title , subtitle , images , background)
        score = curLayout.evaluate()
        img = curLayout.paint()
        img.save('./static/img/target.png')
        return render_to_response('index.html' , {'flag':True , 'score':score})
'''

def getImageFromJson(filename):
    banner = json.load(open(filename , 'r'))
    children = banner['children']
    banner_height = banner['height']
    banner_width = banner['width']
    childrens = []
    for item in children:
        child = layout.Children(item['top'] , item['left'] , item['width'] , item['height'] , item['label'] , item['src'] , item['zIndex'])
        childrens.append(child)

    return layout.Layout(banner_height , banner_width , childrens)


def getImagePalette(palette):
	img = np.zeros((40 , 78*5 , 3))
	for i in range(len(palette)):
		if palette[i] == 'N':
			img[: , 78 * i : 78 * (i +1)] = np.asarray([255 , 255 , 255])
		else:
			img[: , 78 * i : 78 * (i +1)] = np.asarray(palette[i])
	return img

def recolor(request):
    if request.method == 'POST':
        json_name = request.POST['case']
        banner = getImageFromJson(os.path.join('./case' , json_name + '.json'))
        #banner = getImageFromJson('0073_1760.json')
        target = banner.paint()
        target.save('./static/img/target.png')

        palette_gen = request.POST['paletteGen']
        render_op = request.POST['render']
        # iseva = request.POST['iseva']
        keys = request.POST.keys()
        iseva = False
        if 'iseva' in keys:
            iseva = True
        # assert False
        orig_score = banner.evaluate()
        # change background and text
        childrens = banner.childrens
        background = childrens[0]
        element = background
        for item in childrens:
            if item.category == 'background':
                background = item
            elif item.category == 'element':
                element = item


        
        url_elem = 'http:' + element.src
        elem = io.imread(url_elem)
        mask = np.ones(elem[:,:,0].shape) * 255
        if elem.shape[2] == 4:
            mask = elem[:,:,3]
            elem = color.rgba2rgb(elem)
            elem = img_as_ubyte(elem)

        elem_color = getProductColor(elem , mask)
        print elem_color
        elem_palette_img = getImagePalette(elem_color)
        scipy.misc.imsave('./static/img/elem_palette.jpg' , elem_palette_img)

        
        pal = palettes.Palette(elem_color)
        
        if palette_gen == '0':
            palette = pal.getRecColorFromColormind()
        else:
            pass
        # palette = [[49,47,49],[91,83,81],[133,155,143],[226,209,167],[235,198,126]]
        score = pal.palette_score(palette)
        scipy.misc.imsave('./static/img/palette.jpg' , getImagePalette(palette))

        contrasts = []
        order = []
        scores = []
        for item in permutations(range(2 , 5) , 2):
            index_bak , index_text = item
            bak_l , _ , _ = colormap.rgb2lab(palette[index_bak][0] , palette[index_bak][1] , palette[index_bak][2])
            text_l , _, _ = colormap.rgb2lab(palette[index_text][0] , palette[index_text][1] , palette[index_text][2])
            contrast = abs(bak_l - text_l)
            contrasts.append(contrast)
            #order.append(item)
            banner.childChange(bak_change(background , palette[index_bak:index_bak + 1] , render_op))

            for item in childrens:
                print item.category
                print item.filename
                if 'title' in item.category:
                    # banner.childChange(title_change(item , palette[index_text:index_text + 1]))
                    # print 'adasd'
                    if color_select(item):
                        banner.childChange(title_change(item , palette[index_text:index_text + 1]))
                    else:
                        banner.childChange(bak_change(item , palette[index_text:index_text + 1] , render_op))
            img = banner.paint()
            if iseva:
                scores.append(banner.evaluate())

            img.save(os.path.join('./static/img/' , str(index_bak) + '_' + str(index_text) + '.png'))
            order.append(str(index_bak) + '_' + str(index_text) + '.png')

        print contrasts
        print order
        # print score , scores
        print score
        if iseva:
            res = zip(order , scores)
        else:
            res = order

        return render_to_response('index.html' , {'flag':True , 'pics':res , 'score':score , 'iseval':iseva , 'orgin_score':orig_score})


# palette format [[r , g , b]]

def bak_change(background , palette , op):
    color_change = option.Option(background , palette)
    if op == '0':
    	color_change.PaletteBased_Recolor()
    else:
    	color_change.LinearMapping_Recolor()
    return background.zIndex

def color_select(child):
    res = False
    url = 'http:' + child.src
    im = io.imread(url)
    if im.shape[2] == 4:
        im = color.rgba2rgb(im)
    km = KMeans(n_clusters = 2 , random_state = 170)
    km.fit(im.reshape(-1 , 3))
    if len(np.unique(km.labels_)) > 1:
        res = False
    else:
        res = True
    return res


def title_change(title , palette):
    url = 'http:' + title.src
    text = io.imread(url)
    # print text[0,0]
    text[:,:,0] , text[:,:,1] , text[:,:,2] = palette[0]
    # print text[0 , 0]
    io.imsave(title.filename , text)
    return title.zIndex

def color_option(request):
    if request.method == 'POST':
        img = request.FILES['image']
        with open('image_option.png' , 'wb+') as des:
            des.write(img.read())
        palette = request.POST['palettes']
        #print type(palette)
        palette = [int(item) for item in palette.split(',')]
        
        index = 0
        palette_list = []
        while(index < len(palette)):
            palette_list.append(palette[index:index + 3])
            index = index +3

        background_type = int(request.POST['background_type'])
        print palette_list
        background = layout.Backgsround('image_option.png' , background_type)
        color_change = option.Option(background , palette_list)
    return render_to_response('index.html',{'flag':False})


if __name__ == '__main__':
    recolor()