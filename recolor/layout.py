#coding=utf-8

from enum import Enum
from PIL import Image , ImageDraw , ImageFont
import colorize
import numpy as np
import math
import requests
from io import BytesIO
from skimage import io , color , img_as_ubyte
from sklearn.cluster import KMeans
class Background_type(Enum):
    single , dual , gradient , complex = range(4)

class Background:
	def __init__(self , background_image , background_type):
		self.background_image = background_image
		self.background_type = background_type

class Text:
	def __init__(self , context , font , color):
		self.context = context
		self.font = font
		self.color = color

#import textwrap
# 文本换行 , 假设行间距为0
def getWrapSize(text , font_size ,  line_space = 0 , font_type = 'msyh.ttf'):
    font = ImageFont.truetype(font_type , font_size)
    #lines = textwrap.wrap(text , width = max_width)
    lines = text

    width = 0
    height = 0

    for line in lines:
        w , h = font.getsize(line)
        if width < w:
            width = w
        height += (h + line_space)


    return (width , height - line_space)

# 这里的text是分行之后的列表的第一项
def getFontSize(area , text):
    font_size = 1
    font = ImageFont.truetype('msyh.ttf' , font_size)
    fontPixel= font.getsize(text[0])
    #fontArea = fontPixel[0] * fontPixel[1] * 1.0

    while fontPixel[0] <= area[0] and fontPixel[1] <= area[1]:
        font_size += 1
        font = ImageFont.truetype('msyh.ttf' , font_size)
        fontPixel= font.getsize(text)
        fontArea = fontPixel[0] * fontPixel[1] * 1.0

    font_size -= 1
    return font_size


# image = Image.open(images)
def getImageSize(image , area):
    w , h = area['image']
    width , height = image.size

    ratio = width * 1.0 / height

    # 假设宽度设置为w
    h_size = w * 1.0 / ratio
    if h_size > h:
        w_size = h * ratio
        return (w_size , h)
    else:
        return (w , h_size)

def get_elem_color(img):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    im = np.array(img)
    km = KMeans(n_clusters=1 , random_state = 170)
    km.fit(im.reshape(-1 , 3))
    return km.cluster_centers_.flatten()


class Children:
    def __init__(self , top , left , width , height , category , src , zIndex , isChange = False):
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        self.category = category
        self.src = src 
        self.zIndex = zIndex
        self.isChange = isChange
        self.filename = self.src.split('/')[-1]

        if self.isChange:
            url = 'http:' + self.src
            r = requests.get(url)
            with open(self.filename , 'wb') as f:
                f.write(r.content)


class Layout:
    def __init__(self , height , width , childrens):
        self.height = height
        self.width = width
        self.childrens = childrens

        #只需要告知哪个类型的元素改变就好了，剩下的就交给paint
    def childChange(self , zIndex):
        index = 0
        # print 'category',category
        for item in self.childrens:
            if item.zIndex == zIndex:
                break
            index +=1
        print index
        self.childrens[index].isChange = True


    def paint(self):
        img = Image.new('RGBA' , (self.width , self.height))
        self.childrens.sort(lambda a,b:a.zIndex-b.zIndex)
        for item in self.childrens:
            if item.isChange:
                elem = Image.open(item.filename)
            else:
                url = 'http:' + item.src
                response = requests.get(url)
                elem = Image.open(BytesIO(response.content))
            elem = elem.resize((item.width , item.height) , Image.ANTIALIAS)
            if elem.mode == 'RGBA':
                img.paste(elem , (item.left , item.top) , mask = elem)
            else:
                img.paste(elem , (item.left , item.top))

        return img

    def evaluate(self):
        img = self.paint()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w , h = img.size
        ratio = w * 1.0 / h
        tmp = img
        img = img.resize((int(ratio * 128) , 128) , Image.ANTIALIAS)
        im = np.array(img)
        density_unary = colorize.unary(colorize.getUnaryFea(im))
        bak_color , bak_color_label , labels = colorize.getBackgroundColor(im)
        bak_area = colorize.getColorArea(labels , bak_color_label)
        #print self.title.color , bak_color
        title = None
        for item in self.childrens:
            if 'title' in item.category:
                title = item
                if item.isChange:
                    title_elem = Image.open(item.filename)
                else:
                    url = 'http:' + item.src
                    response = requests.get(url)
                    title_elem = Image.open(BytesIO(response.content))
                break;

        title_color = get_elem_color(title_elem)
        # title_color = np.array([self.title.color[0] , self.title.color[1] , self.title.color[2]])
        density_pairwise = colorize.pairwise(colorize.getPairwiseFea(im , title_color , bak_color))
        x , y = title.left , title.top
        w , h = title.width , title.height
        img_font = tmp.crop((x , y , x + w , y + h))
        contrast_strength = colorize.getStrength(np.array(img_font))

        bak_light , _ , _ = colorize.rgbTolab(bak_color[0] , bak_color[1] , bak_color[2])
        font_light , _ , _ = colorize.rgbTolab(title_color[0] , title_color[1] , title_color[2])
        contrast_value = abs(bak_light - font_light)
        return math.log(density_unary(bak_light) * bak_area) +  math.log(density_pairwise(contrast_value) * contrast_strength)



'''
class Layout:
    def __init__(self , pos , area , size , title , subtitle , images , background):
        self.pos = pos 
        self.area = area
        self.size = size 
        self.title = title
        self.subtitle = subtitle
        self.images = images
        self.background = background
    def paintText(self , detail):
        if detail.context == "":
            return Image.new("RGBA" , (1 , 1))
        text = detail.context.split('\r\n')
        lines = len(text)
        w , h = self.area['title']
        line_size = h / lines
        font_size = getFontSize((w , line_size) , text[0])
        text_size = getWrapSize(text , font_size , line_space = 0 , font_type = detail.font)
        img = Image.new("RGBA" , text_size)
        draw = ImageDraw.ImageDraw(img , "RGBA")
        font = ImageFont.truetype(detail.font , font_size)
        y_text = 0
        for item in text:
            width , height = font.getsize(item)
            draw.text((0 , y_text) , item , fill = detail.color , font = font)
            y_text += height
        return img

    def paintPic(self):
        for item in self.images:
            img = Image.open(item)
            image_size = getImageSize(img , self.area)
            image_w , image_h = image_size

            img = img.resize((int(image_w) , int(image_h)) , Image.ANTIALIAS)
        return img

    def paintBackground(self):
        back = Image.open(self.background.background_image)
        return back.resize(self.size , Image.ANTIALIAS)

    def paint(self):
        img = Image.new('RGBA' , self.size , (255 , 255 , 255))
        Draw = ImageDraw.ImageDraw(img , "RGBA")
        background = self.paintBackground()
        img.paste(background , (0 , 0))

        pic = self.paintPic()
        img.paste(pic , self.pos['image'] , mask = pic)

        title = self.paintText(self.title)
        img.paste(title , self.pos['title'] , mask = title)

        subtitle = self.paintText(self.subtitle)
        img.paste(subtitle , self.pos['subtitle'] , mask = subtitle)
        return img

    def evaluate(self):
        img = self.paint()
        if img.mode != 'RGB':
            img = img.convert('RGB')

        im = np.array(img)
        density_unary = colorize.unary(colorize.getUnaryFea(im))
        bak_color , bak_color_label , labels = colorize.getBackgroundColor(im)
        bak_area = colorize.getColorArea(labels , bak_color_label)
        #print self.title.color , bak_color
        title_color = np.array([self.title.color[0] , self.title.color[1] , self.title.color[2]])
        density_pairwise = colorize.pairwise(colorize.getPairwiseFea(im , title_color , bak_color))
        x , y = self.pos['title']
        w , h = self.area['title']
        img_font = img.crop((x , y , x + w , y + h))
        contrast_strength = colorize.getStrength(np.array(img_font))

        bak_light , _ , _ = colorize.rgbTolab(bak_color[0] , bak_color[1] , bak_color[2])
        font_light , _ , _ = colorize.rgbTolab(self.title.color[0] , self.title.color[1] , self.title.color[2])
        contrast_value = abs(bak_light - font_light)
        return math.log(density_unary(bak_light) * bak_area) + math.log(density_pairwise(contrast_value) * contrast_strength)

'''
















