# -*- coding: utf-8 -*-
from PIL import Image ,ImageDraw,ImageOps
from color2 import *
import math
import numpy as np 
import hsl
import time

class Mapping(object):
	def __init__(self,img,palette,num):
		if img.mode!='RGB':
			img=img.convert('RGB')
		self.img = img
		self.img_copy = img.copy()
		self.dataArray = list(img.getdata()) 
		self.data_lab=[]
		for i in range(len(self.dataArray)):
			self.data_lab.append(rgb2lab(self.dataArray[i]))
		self.palette=palette
		self.gradation_num=num
		self.bin_range = 16;
		self.bin_size = 256 / self.bin_range;

	# line1 is the origin regression line
	# line2 is the template regression line
	def getRotateMat(self,line1,line2):
		angle1=np.arctan(line1[0])
		angle2=np.arctan(line2[0])
		angle=angle2-angle1
		mat=np.zeros((3,3))
		mat[0][0]=np.cos(angle)
		mat[0][1]=-np.sin(angle)
		mat[0][2]=0
		mat[1][0]=np.sin(angle)
		mat[1][1]=np.cos(angle)
		mat[1][2]=0
		mat[2][0]=mat[2][1]=0
		mat[2][2]=1
		return mat
	def getRotateMat_angle(self,angle1,angle2):
		angle=angle2-angle1
		mat=np.zeros((3,3))
		mat[0][0]=np.cos(angle)
		mat[0][1]=-np.sin(angle)
		mat[0][2]=0
		mat[1][0]=np.sin(angle)
		mat[1][1]=np.cos(angle)
		mat[1][2]=0
		mat[2][0]=mat[2][1]=0
		mat[2][2]=1
		return mat
	# y=alpha*x+beta
	# mark为True，则是将直线平移到原点
	# mark为False，则是将直线从原点平移到所在位置
	def getTranslateMat(self,line,mark):
		alpha=line[0]
		beta=line[1]
		if mark==True:
			if abs(alpha)<=1:
				mat=[[1,0,0],[0,1,-beta],[0,0,1]]
			else:
				mat=[[1,0,beta*1.0/alpha],[0,1,0],[0,0,1]]
		else:
			if abs(alpha)<=1:
				mat=[[1,0,0],[0,1,beta],[0,0,1]]
			else:
				mat=[[1,0,-beta*1.0/alpha],[0,1,0],[0,0,1]]
		return mat
	# numpy其实有np.linspace(x1,x2,n)函数...
	def linespace(self,x1,x2,n):
		delta=(x2-x1)*1.0/(n-1)
		print delta
		vector=[]
		for i in range(0,n-1):
			vector.append(x1+delta*i)
		vector.append(x2)
		return vector
	# n是渐变的层数
	def gradation2Color(self,color1,color2,n):
		r1,g1,b1=color1
		r2,g2,b2=color2
		vec1=self.linespace(r1,r2,n)
		vec2=self.linespace(g1,g2,n)
		vec3=self.linespace(b1,b2,n)
		res=[]
		for i in range(n):
			res.append([vec1[i],vec2[i],vec3[i]])
		return res

	def gradationPalette(self,palette,n):
		l=len(palette)
		res=[]
		for i in range(0,l-1):
			v=self.gradation2Color(palette[i],palette[i+1],n)
			if i==0:
				res.append(v[0])
			for j in range(1,n):
				res.append(v[j])
		return res
	def linearFit(self,X,Y):
		z1=np.polyfit(X,Y,1)
		return [z1[0],z1[1]]
	def lab2line(self,colors):
		a=[x[1] for x in colors]
		b=[x[2] for x in colors]
		line=self.linearFit(a,b)
		return line
	def recolor(self):
		line_origin=self.lab2line(self.data_lab)
		gradation_rgb=self.gradationPalette(self.palette,self.gradation_num)
		self.gradation_rgb=gradation_rgb
		gradation_lab=[]
		for i in range(len(gradation_rgb)):
			gradation_lab.append(rgb2lab(gradation_rgb[i]))
		line_template=self.lab2line(gradation_lab)
		Rmat=self.getRotateMat(line_origin,line_template)
		Tmat1=self.getTranslateMat(line_origin,True)
		Tmat2=self.getTranslateMat(line_template,False)
		
		pix=self.img_copy.load()
		width=self.img_copy.size[0]
		height=self.img_copy.size[1]

		mat_X=[]
		L_save=[]
		for x in range(width):
			print x####################################################################################
			for y in range(height):
				#print pix[x,y]
				R,G,B=pix[x,y]  # x is horizontal, y is vertical
				Lab=rgb2lab([R,G,B])
				L_save.append(Lab[0])
				mat_X.append([Lab[1],Lab[2],1])
		X_origin=np.asarray(mat_X)

		X_origin=X_origin.transpose()
		X_map=np.dot(Tmat1,X_origin)
		X_map=np.dot(Rmat,X_map)
		X_map=np.dot(Tmat2,X_map)
		X_map=X_map.transpose()
		# 为什么X_map每一行的第三个是0？不应该是1么
		for x in range(width):
			for y in range(height):
				idx=x*height+y
				Lab=[L_save[idx],int(X_map[idx][0]),int(X_map[idx][1])]
				rgb=lab2rgb(Lab)
				pix[x,y]=(rgb[0],rgb[1],rgb[2])
		return self.img_copy
	# y=alpha*x+beta
	# 平移是根据color到line的垂直距离的水平和竖直方向来求得
	def getSingleColorTmat(self,line,color):
		Lab=rgb2lab(color)
		a=Lab[1]
		b=Lab[2]
		alpha=line[0]
		beta=line[1]
		l=abs((alpha*a-b+beta)*1.0/(math.sqrt(alpha*alpha+1)))
		print 'l',l
		angle=np.arctan(abs(alpha))
		print 'line',line
		print 'angle',angle*180/math.pi
		delta_x=l*np.sin(angle)
		delta_y=l*np.cos(angle)
		if alpha>=0:
			b2=alpha*a+beta
			# color在直线下方
			if b2>=b:
				matrix=[[1,0,delta_x],[0,1,-delta_y],[0,0,1]]
			# color在直线上方
			else:
				matrix=[[1,0,-delta_x],[0,1,delta_y],[0,0,1]]
		else:
			b2=alpha*a+beta
			if b2>=b:
				matrix=[[1,0,-delta_x],[0,1,-delta_y],[0,0,1]]
			else:
				matrix=[[1,0,delta_x],[0,1,delta_y],[0,0,1]]
		return matrix
	def getMainColor(self):
		count=np.zeros((self.bin_range,self.bin_range,self.bin_range))
		for i in range(len(self.dataArray)):
			R,G,B=self.dataArray[i]
			ri=R/self.bin_size
			gi=G/self.bin_size
			bi=B/self.bin_size
			count[ri][gi][bi]+=1
		maxcnt=0
		for i in range(self.bin_range):
			for j in range(self.bin_range):
				for k in range(self.bin_range):
					if count[i][j][k]>maxcnt:
						mainColor=[(i+0.5)*self.bin_size,(j+0.5)*self.bin_size,(k+0.5)*self.bin_size]
						maxcnt=count[i][j][k]
		return mainColor
	# 平移矩阵是根据主色调和color之间的ab差距来求得
	def getSingleColorTmat2(self,color):
		Lab=rgb2lab(color)
		mainColor=self.getMainColor()
		mainLab=rgb2lab(mainColor)
		print 'maincolor',mainColor,mainLab
		print 'palette',color,Lab
		delta_x=Lab[1]-mainLab[1]
		delta_y=Lab[2]-mainLab[2]
		matrix=[[1,0,delta_x],[0,1,delta_y],[0,0,1]]
		return matrix

	def getSingleColorTmat3(self,line,mark):
		mainColor=self.getMainColor()
		mainLab=rgb2lab(mainColor)
		alpha1=line[0]
		beta1=line[1]
		if alpha1!=0:
			alpha2=-1*1.0/alpha1
			beta2=mainLab[2]-alpha2*mainLab[1]
			p_x=1.0*(beta2-beta1)/(alpha1-alpha2)
			p_y=alpha1*p_x+beta1
			if mark==True:
				matrix=[[1,0,-p_x],[0,1,-p_y],[0,0,1]]  # 将line从点p平移到原点
			else:
				matrix=[[1,0,p_x],[0,1,p_y],[0,0,1]]  # 将line从原点平移到点p
		else:
			p_x=mainLab[1]
			p_y= beta1
			if mark==True:
				matrix=[[1,0,-p_x],[0,1,-p_y],[0,0,1]]  # 将line从点p平移到原点
			else:
				matrix=[[1,0,p_x],[0,1,p_y],[0,0,1]]  # 将line从原点平移到点p
		return matrix,[p_x,p_y]

	# 如果palette只有一个颜色，平移是根据该颜色到直线的距离来决定
	def recolor_SingleColor(self):
		line_origin=self.lab2line(self.data_lab)
		#gradation_rgb=self.gradationPalette(self.palette,self.gradation_num)
		Tmat=self.getSingleColorTmat(line_origin,self.palette[0])
		print 'Tmat1',Tmat
		#Tmat=self.getSingleColorTmat2(self.palette[0])
		## 注意由于在main里面是同一个pp，所以应该立即先保存，否则img_recolor2改动后，这个也会随之改动
		self.img_copy = img.copy()
		pix=self.img_copy.load()
		width=self.img_copy.size[0]
		height=self.img_copy.size[1]

		mat_X=[]
		L_save=[]
		for x in range(width):
			#print x####################################################################################
			for y in range(height):
				#print pix[x,y]
				R,G,B=pix[x,y]  # x is horizontal, y is vertical
				Lab=rgb2lab([R,G,B])
				L_save.append(Lab[0])
				mat_X.append([Lab[1],Lab[2],1])
		X_origin=np.asarray(mat_X)

		X_origin=X_origin.transpose()
		X_map=np.dot(Tmat,X_origin)
		X_map=X_map.transpose()
		# 为什么X_map每一行的第三个是0？不应该是1么
		for x in range(width):
			for y in range(height):
				idx=x*height+y
				Lab=[L_save[idx],int(X_map[idx][0]),int(X_map[idx][1])]
				rgb=lab2rgb(Lab)
				pix[x,y]=(rgb[0],rgb[1],rgb[2])
		return self.img_copy
		# 如果palette只有一个颜色，平移是根据主色调与palette之间的ab差距
	def recolor_SingleColor2(self):
		line_origin=self.lab2line(self.data_lab)
		#gradation_rgb=self.gradationPalette(self.palette,self.gradation_num)
		#Tmat=self.getSingleColorTmat(line_origin,self.palette[0])
		Tmat=self.getSingleColorTmat2(self.palette[0])
		print 'Tmat2',Tmat
		self.img_copy = img.copy()
		pix=self.img_copy.load()
		width=self.img_copy.size[0]
		height=self.img_copy.size[1]

		mat_X=[]
		L_save=[]
		for x in range(width):
			#print x####################################################################################
			for y in range(height):
				#print pix[x,y]
				R,G,B=pix[x,y]  # x is horizontal, y is vertical
				Lab=rgb2lab([R,G,B])
				L_save.append(Lab[0])
				mat_X.append([Lab[1],Lab[2],1])
		X_origin=np.asarray(mat_X)

		X_origin=X_origin.transpose()
		X_map=np.dot(Tmat,X_origin)
		X_map=X_map.transpose()
		# 为什么X_map每一行的第三个是0？不应该是1么
		for x in range(width):
			for y in range(height):
				idx=x*height+y
				Lab=[L_save[idx],int(X_map[idx][0]),int(X_map[idx][1])]
				rgb=lab2rgb(Lab)
				pix[x,y]=(rgb[0],rgb[1],rgb[2])
		return self.img_copy
	# 计算mainColor到line1最近的点p1（即正交的交点），p2为palette的点，对line1进行旋转操作使得与p1,p2垂直
	# （首先得将line1平移到原点，再旋转，再平移回到p1点），再将line1平移到p2
	def recolor_SingleColor3(self):
		line_origin=self.lab2line(self.data_lab)
		#gradation_rgb=self.gradationPalette(self.palette,self.gradation_num)
		#self.gradation_rgb=gradation_rgb
		#gradation_lab=[]
		#for i in range(len(gradation_rgb)):
		#	gradation_lab.append(rgb2lab(gradation_rgb[i]))
		#line_template=self.lab2line(gradation_lab)
		Tmat1,point1=self.getSingleColorTmat3(line_origin,True)
		Tmat2,point1=self.getSingleColorTmat3(line_origin,False)
		Lab=rgb2lab(self.palette[0])
		point2=[Lab[1],Lab[2]]
		if point1[0]!=point2[0]:
			alpha=(point2[1]-point1[1])*1.0/(point2[0]-point1[0])
			if alpha!=0:
				alpha2=-1.0/alpha
				beta2=point1[1]-alpha2*point1[0]
				line2=[alpha2,beta2]
				Rmat=self.getRotateMat(line_origin,line2)
			else:
				angle2=0.5*math.pi
				angle1=np.arctan(line_origin[0])
				Rmat=self.getRotateMat_angle(angle1,angle2)
		else:
			angle2=0
			angle1=np.arctan(line_origin[0])
			Rmat=self.getRotateMat_angle(angle1,angle2)
		delta_x=point2[0]-point1[0]
		delta_y=point2[1]-point1[1]
		Tmat3=[[1,0,delta_x],[0,1,delta_y],[0,0,1]]
		print 'line1',line_origin
		print 'mainLab',rgb2lab(self.getMainColor())
		print 'palette',Lab
		print 'Tmat1',Tmat1
		print 'Rmat',Rmat
		print 'Tmat2',Tmat2
		print 'Tmat3',Tmat3

		self.img_copy = self.img.copy()
		pix=self.img_copy.load()
		width=self.img_copy.size[0]
		height=self.img_copy.size[1]

		mat_X=[]
		L_save=[]
		for x in range(width):
			#print x####################################################################################
			for y in range(height):
				#print pix[x,y]
				R,G,B=pix[x,y]  # x is horizontal, y is vertical
				Lab=rgb2lab([R,G,B])
				L_save.append(Lab[0])
				mat_X.append([Lab[1],Lab[2],1])
		X_origin=np.asarray(mat_X)

		X_origin=X_origin.transpose()
		X_map=np.dot(Tmat1,X_origin)
		X_map=np.dot(Rmat,X_map)
		X_map=np.dot(Tmat2,X_map)
		X_map=np.dot(Tmat3,X_map)
		X_map=X_map.transpose()
		# 为什么X_map每一行的第三个是0？不应该是1么
		for x in range(width):
			for y in range(height):
				idx=x*height+y
				Lab=[L_save[idx],int(X_map[idx][0]),int(X_map[idx][1])]
				rgb=lab2rgb(Lab)
				pix[x,y]=(rgb[0],rgb[1],rgb[2])
		return self.img_copy
def linespace(x1,x2,n):
	delta=(x2-x1)*1.0/(n-1)
	print delta
	vector=[]
	for i in range(0,n-1):
		vector.append(x1+delta*i)
	vector.append(x2)
	return vector
def save_theme(filename,colors,width,height):
	img=Image.new('RGB',[width,height])
	img_pix=img.load()
	width=img.size[0]
	height=img.size[1]
	k=len(colors)
	xlen=width/k
	for i in range(0,k):
		for x in range(xlen*i,min(xlen*i+xlen,width)):
			for y in range(height):
				R=int(colors[i][0])
				G=int(colors[i][1])
				B=int(colors[i][2])
				img_pix[x,y]=(R,G,B)
	# 剩余的就补最后一个颜色
	for x in range(xlen*(k-1)+xlen,width):
		for y in range(height):
			R=int(colors[k-1][0])
			G=int(colors[k-1][1])
			B=int(colors[k-1][2])
			img_pix[x,y]=(R,G,B)
	img.save(filename)
#img = Image.open(filename)
	#img.show()
#	if img.mode != 'RGB':
#		img = img.convert('RGB')
# Lab1=[50,40,50]
# print lab2rgb(Lab1)
# Lab2=[50,70,70]
# print lab2rgb(Lab2)
# filename='input/map4.png'
# colors=[]
# #colors2.append([0,0,0])  ### 第一个忽略
# file = open('colors.txt','r');
# for line in file:
# 	line=line.strip('\n') # remove '\n' at the end of line
# 	#a=line.split(" ")[0]
# 	#b=line.split(" ")[1]
# 	#c=line.split(" ")[2]
# 	a,b,c=map(int,line.split())
# 	colors.append([a,b,c])
# 	print [a,b,c]
# file.close();

# start=time.time()

# img = Image.open(filename)
# if img.mode != 'RGB':
# 		img = img.convert('RGB')
# pp=Mapping(img,colors,20)
# test_id=8
# if len(colors)>1:
# 	img_recolor=pp.recolor()
	
# 	save_theme('res/'+str(test_id)+'_gradation.png',pp.gradation_rgb,1000,100)
# 	img.save('res/'+str(test_id)+'_img.png')
# 	img_recolor.save('res/'+str(test_id)+'_res.png')
# else:
# 	save_theme('res/'+str(test_id)+'_gradation.png',pp.palette,1000,100)
# 	img.save('res/'+str(test_id)+'_img.png')

# 	img_recolor=pp.recolor_SingleColor()
# 	img_recolor.save('res/'+str(test_id)+'_res_l2p.png')

# 	img_recolor2=pp.recolor_SingleColor2()  
# 	img_recolor2.save('res/'+str(test_id)+'_res_p2p.png')

# 	img_recolor3=pp.recolor_SingleColor3()  
# 	img_recolor3.save('res/'+str(test_id)+'_res_TRTT.png')

# end=time.time()
# print 'Time is %0.2f seconds',(end-start)