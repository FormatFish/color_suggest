#coding=utf-8
import layout
import scipy.misc
from sklearn.cluster import KMeans
from PIL import Image ,ImageDraw,ImageOps
from color2 import *
import rendering
import math
import numpy as np 
import linear_mapping
from skimage import io , color , img_as_ubyte
# class Background:
#     def __init__(self,background_image,background_type):
#         self.background_image=background_image
#         self.background_type=background_type
class Option:
    def __init__(self , background , palette):
        self.background_image = background.filename
        # self.background_type = background.background_type
        self.palette = palette
        #self.bak = scipy.misc.imread(self.background_image , mode='RGB')
        

        url = 'http:' + background.src
        bak = io.imread(url)
        io.imsave(self.background_image , bak)
        self.mask = bak[:,:,3]
        if bak.shape[2] == 4:
            bak = color.rgba2rgb(bak)
            bak = img_as_ubyte(bak)
        self.bak = bak
        self.rows , self.cols , self.channel = self.bak.shape
        self.k = 5
        self.palette_len=len(self.palette)

        # if self.palette_len==1:
        #     self.HueColorChange()
        # else:
        #     self.ColorChange()
        #if self.background_type == layout.Background_type.single:
        #    print 'hhhhhhh'
        #    self.singleColorChange()
        #elif self.background_type == layout.Background_type.dual:
        #    self.textureColorChange()
        #elif self.background_type == layout.Background_type.complex:
        #    print 'helllll'
        #    self.complexColorChange()

    # paper 2
    def LinearMapping_Recolor(self):
        img= Image.open(self.background_image)
        mask = np.array(img.split()[3])
        if img.mode != 'RGB':
            img = img.convert('RGB') 
        lmap=linear_mapping.Mapping(img,self.palette,20)
        if len(self.palette)>1:
            img_recolor=lmap.recolor()
        else:
            img_recolor=lmap.recolor_SingleColor3() 
        pix=img_recolor.load()
        for row in range(self.rows):
            for col in range(self.cols):
                RGB=pix[col,row]
                self.bak[row , col] = [RGB[0],RGB[1],RGB[2]]

        self.bak = np.dstack((self.bak , mask))
        scipy.misc.imsave(self.background_image , self.bak)

    # paper 1
    def PaletteBased_Recolor(self):
        if self.palette_len==1:
            self.HueColorChange()
        else:
            self.ColorChange()
    def HueColorChange(self):
        img = Image.open(self.background_image)
        mask = np.array(img.split()[3])
        if img.mode != 'RGB':
            img = img.convert('RGB')   

        pp=rendering.Rendering(img,self.k)
        color=self.palette[0]
        img_hue=pp.SingleChangeHue(color)
        pix=img_hue.load()
        for row in range(self.rows):
            for col in range(self.cols):
                RGB=pix[col,row]
                self.bak[row , col] = [RGB[0],RGB[1],RGB[2]]
        self.bak = np.dstack((self.bak , mask))
        scipy.misc.imsave(self.background_image , self.bak)
        #img_hue.save('res/hue_res1.png')

    def ColorChange(self):
        img = Image.open(self.background_image)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        pp=rendering.Rendering(img,self.k)
        colors2=[]
        colors2.append([0,0,0])
        for i in range(0,len(self.palette)):
            colors2.append(self.palette[i])
        img_rendering=pp.rendering(colors2)
        pix=img_rendering.load()
        for row in range(self.rows):
            for col in range(self.cols):
                RGB=pix[col,row]
                self.bak[row , col] = [RGB[0],RGB[1],RGB[2]]
        scipy.misc.imsave(self.background_image , self.bak)

    '''
    def singleColorChange(self):
        for row in range(self.rows):
            for col in range(self.cols):
                self.bak[row , col] = [self.palette[-3][0] , self.palette[-3][1] , self.palette[-3][2]]
        scipy.misc.imsave(self.background_image , self.bak)

    def textureColorChange(self):
        km = KMeans(n_clusters = self.k , random_state = 170)
        km.fit(self.bak.reshape(-1 , 3))
        labels = km.labels_
        bak_color = [self.palette[-3][0]*1.0/255 , self.palette[-3][1]*1.0/255 , self.palette[-3][2]*1.0/255]
        #print self.palette
        #sub_bak_color = [palette[-2][0]*1.0/255 , palette[-2][1]*1.0/255 , palette[-2][2]*1.0/255]
        sub_bak_color = [self.palette[-2][0]*1.0/255 , self.palette[-2][1]*1.0/255 , self.palette[-2][2]*1.0/255]
        dst = color.label2rgb(labels.reshape(-1 , self.bak.shape[1]) , colors = [bak_color , sub_bak_color])
        scipy.misc.imsave(self.background_image , dst)
    def complexColorChange(self):
        img = Image.open(self.background_image)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        pp=rendering.Rendering(img)
        colors2=[]
        colors2.append([0,0,0])
        for i in range(0,len(self.palette)):
            colors2.append(self.palette[i])
        img_rendering=pp.rendering(colors2,0,False)
        pix=img_rendering.load()
        for row in range(self.rows):
            for col in range(self.cols):
                RGB=pix[col,row]
                self.bak[row , col] = [RGB[0],RGB[1],RGB[2]]
        scipy.misc.imsave(self.background_image , self.bak)
    '''