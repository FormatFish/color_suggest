# -*- coding: utf-8 -*-
from PIL import Image ,ImageDraw,ImageOps
from color2 import *
import math
import numpy as np 
import time
import hsl
class Color_cluster:
    def __init__(self,inputColor):
        self.color = inputColor     # color
        self.count = 0     
        self.idx = -1     
        self.Lab=[]

class Rendering(object):
	"""docstring for ClassName"""
	def __init__(self, img,cluster_k):
		self.bins={}
		self.bin_range = 16;
		self.bin_size = 256 / self.bin_range;
		self.channels = 4;
		if img.mode!='RGB':
			img=img.convert('RGB')
		self.img = img;
		self.img_copy = img.copy()
		self.img_copy2 = img.copy()
		self.dataArray = list(img.getdata()); #img已经转化为RGB，所以数据形式是[(r,g,b),(r,g,b),(),...]
		#self.ctx = ctx;
		self.K = cluster_k;  # the main K colors
		#self.L1=[0]
		#self.L2=[0]
		self.Ls=[] # the lightness of original main colors
		self.Ls_change=[] # the lightness of main colors after changed
		#self.color_list1=[[0 for col in range(3)] for row in range(self.K+1)]  # init 6*3 arrays
		#self.color_list2=[[0 for col in range(3)] for row in range(self.K+1)]
		self.color_list1=[]
		self.color_list2=[]
		self.centers_lab=[]
		self.centers_rgb=[]
		#self.Color =Color()
		for i in range(0,self.bin_range):
			for j in range(0,self.bin_range):
				for k in range(0,self.bin_range):
					tmp=Color_cluster([(i + 0.5) * self.bin_size, (j + 0.5) * self.bin_size, (k + 0.5) * self.bin_size])
					tmp.Lab= rgb2lab(tmp.color)
					self.bins['r'+str(i)+'g'+str(j)+'b'+str(k)]=tmp
		self.palette()
		self.kmeans()

	def palette(self):
		length=len(self.dataArray)
		for i in range(0,length):
			R=self.dataArray[i][0]
			G=self.dataArray[i][1]
			B=self.dataArray[i][2]
			ri=R/self.bin_size
			gi=G/self.bin_size
			bi=B/self.bin_size
			self.bins['r'+str(ri)+'g'+str(gi)+'b'+str(bi)].count+=1

	def distance2(self,c1,c2):
		res=0;
		for i in range(0,len(c1)):
			res+=(c1[i]-c2[i])*(c1[i]-c2[i])
		return res

	def normalize(self,v):
		d=math.sqrt(self.distance2(v,[0,0,0]))
		res=[]
		for i in range(0,len(v)):
			res.append(v[i]/d)
		return res
	def add(self,c1,c2):
		res=[]
		for i in range(0,len(c1)):
			res.append(c1[i]+c2[i])
		return res

	def sub(self,c1,c2):
		res=[]
		for i in range(0,len(c1)):
			res.append(c1[i]-c2[i])
		return res

	def sca_mul(self,c,k):
		res=[]
		for i in range(0,len(c)):
			res.append(c[i]*k)
		return res

	def kmeansFirst(self):
		centers = []  #rgb format
		centers_lab=[]
		centers.append([self.bin_size/2,self.bin_size/2,self.bin_size/2]) #black
		centers_lab.append(rgb2lab(centers[0]))
		bins_copy={}  #dict
		#i is key
		for i in self.bins:
			bins_copy[i]=self.bins[i].count

		for p in range(0,self.K):
			maxc=-1
			for i in bins_copy:
				d2=self.distance2(self.bins[i].Lab,centers_lab[p])
				# the longer the d2 is, the larger the factor is
				factor=1-math.exp(-d2/6400.0)
				bins_copy[i]*=factor
				if bins_copy[i] > maxc:
					maxc=bins_copy[i]
					tmp=[]
					for j in range(0,3):
						tmp.append(self.bins[i].color[j])
			# tmp is the most distant and largest bin
			# each time, we find a tmp as one of clusters
			centers.append(tmp)
			centers_lab.append(rgb2lab(tmp))
		return centers_lab  # black + K colors, so it contains K+1 colors
	# 返回1+K 个颜色，第一个是黑色，可以不用考虑
	def kmeans(self):
		centers=self.kmeansFirst()
		no_change=False
		centers_count=[]
		centers_count.append(float('inf'))
		while not no_change:
			no_change=True
			sum=[]
			for i in range(0,self.K+1):
				sum.append(Color_cluster([0,0,0]))
			for i in range(0,self.bin_range):
				for j in range(0,self.bin_range):
					for k in range(0,self.bin_range):
						tmp=self.bins['r'+str(i)+'g'+str(j)+'b'+str(k)]
						if tmp.count==0:
							continue
						lab=tmp.Lab
						mind=float('inf')
						mini=-1
						for p in range(0,self.K+1):
							d=self.distance2(centers[p],lab)
							if d<mind:
								mind=d
								mini=p
						if mini!=tmp.idx:
							tmp.idx=mini
							no_change=False
						###########m是LAB，而sum[].color是RGB，不过一开始是[0,0,0]，所以没有影响###################
						m=self.sca_mul(tmp.Lab,tmp.count)
						sum[mini].color=self.add(sum[mini].color,m)
						sum[mini].count+=tmp.count

			for i in range(1,self.K+1):
				centers_count.append(sum[i].count)
				if sum[i].count>0:
					for j in range(0,3):
						centers[i][j]=sum[i].color[j]*1.0/sum[i].count
					#centers_count.append(sum[i].count)
						#print centers[i][j],sum[i].color[j],sum[i].count
		centers=self.sort_counts(centers,centers_count)
		for i in range(0,len(centers)):
			L=int(centers[i][0])
			A=int(centers[i][1])
			B=int(centers[i][2])
			self.centers_lab.append([L,A,B])
		centers_rgb=[]
		for i in range(0,self.K+1):
			centers_rgb.append(lab2rgb(centers[i]))
		self.centers_rgb=centers_rgb
		return centers_rgb

	# sort the cluster colors according to their lightness
	def sort(self,colors):
		l=len(colors)
		for i in range(1,l)[::-1]:
			for j in range(0,i):
				if colors[j][0]>colors[j+1][0]:
					tmp=colors[j]
					colors[j]=colors[j+1]
					colors[j+1]=tmp
		return colors
	def sort_lightness(self,ligntness):
		l=len(ligntness)
		for i in range(1,l)[::-1]:
			for j in range(0,i):
				if ligntness[j]>ligntness[j+1]:
					tmp=ligntness[j]
					ligntness[j]=ligntness[j+1]
					ligntness[j+1]=tmp
		return ligntness
	# sort the cluster colors according to their counts, descending
	def sort_counts(self,colors,counts):
		l=len(colors)
		for i in range(1,l)[::-1]:
			for j in range(0,i):
				if counts[j]<counts[j+1]:
					tmp=colors[j]
					colors[j]=colors[j+1]
					colors[j+1]=tmp
					tmp2=counts[j]
					counts[j]=counts[j+1]
					counts[j+1]=tmp2
		return colors
	# colors1 and colors2 are Lab form
	def colorTransform(self,colors1,colors2):
		self.L1=[0]
		self.L2=[0]
		for i in range(1,len(colors1)):
			self.L1.append(colors1[i][0])
			self.L2.append(colors2[i][0])
		self.L1.append(100)
		self.L2.append(100)
		self.L1=self.sort_lightness(self.L1)
		self.L2=self.sort_lightness(self.L2)
		print 'L1',self.L1 ###################################
		print 'L2',self.L2 ##################################
		#l=len(self.dataArray)
		pix=self.img_copy.load()  #load() can be used to modify the value of pixel

		cs1=[]
		cs2=[]
		k=0
		for i in range(1,len(colors2)):
			#if colors2[i]!=False:  # if the colors2[i]==False, means it is not be modified
			#if distance2(colors1[i],colors2[i])>1:
			cs1.append(colors1[i])
			cs2.append(colors2[i])
			k+=1
		print 'cs1',cs1
		print 'cs2',cs2
		##### getSigma 应该是所有聚类出来的颜色的平均值
		#self.sigma=self.getSigma(colors1)
		self.sigma=self.getSigma(self.centers_lab)
		self.lambdas=self.getLambda(cs1)
		#print 'lambdas',len(self.lambdas[0]),len(self.lambdas)
		#print '[0][0]',self.lambdas[0][0],'[0][1]',self.lambdas[0][1]
		#print self.lambdas #########################################################################
		width=self.img_copy.size[0]
		height=self.img_copy.size[1]

		vis=np.zeros((256,256,256))
		color_transform=np.zeros((256,256,256,3))
		for x in range(width):
			print x####################################################################################
			for y in range(height):
				#print pix[x,y]
				R,G,B=pix[x,y]  # x is horizontal, y is vertical
				# 已经计算过得就不需要重复计算了
				if vis[R][G][B]==1:
					rgb=color_transform[R][G][B]
					pix[x,y]=(int(rgb[0]),int(rgb[1]),int(rgb[2]))
					continue

				Lab=rgb2lab([R,G,B])
				out_lab=[0,0,0]
				L=self.colorTransformSingleL(Lab[0])
				for p in range(0,k):
					#print 'cs1,cs2',[cs1[p][1],cs1[p][2]],[cs2[p][1],cs2[p][2]]  #######################
					v=self.colorTransformSingleAB([cs1[p][1],cs1[p][2]],[cs2[p][1],cs2[p][2]],Lab[0],Lab)
					v[0]=L
					#print 'v1',v,lab2rgb(v) ####################################################
					omega=self.omega(cs1,Lab,p) # omega is the weight
					#omega=1
					####################### omega有绝对值很小的，所以导致v也很小，导致最后的out_lab几乎为(0,0,0)，使得rgb值也为(0,0,0) ##############################################################################
					####################### omega绝对值很小，是因为omega函数里phi计算的值很小，phi函数计算的值很小，是因为参数r的值（即cs1[j]和Lab的距离）很大，或者是sigma值较小####################################
					####################### sigma值较小，是因为计算getSigma函数，没有计算所有聚类出来的K个颜色的平均距离，而只是计算了转换的cs1的平均距离####################################
					#if x==32 and y==39:
					#	omega=self.omega(cs1,Lab,p)
					#	print omega,'v1',v
					v=self.sca_mul(v,omega)
					#if x==32 and y==39:
					#	print 'v2',v
					#print v
					#print 'v2',v,lab2rgb(v) ####################################################
					out_lab=self.add(out_lab,v) # the sum of RBF
				#print out_lab   ###########################################
				out_rgb=lab2rgb(out_lab)
				#if x==32 and y==39:
				#	print out_rgb,out_lab
				#print out_rgb ##################################
				#if out_rgb[0]<=0 or out_rgb[1]<=0 or out_rgb[2]<=0:
				#	print 'a1',out_rgb,out_lab
				out_rgb[0]=max(min(out_rgb[0],255),0)
				out_rgb[1]=max(min(out_rgb[1],255),0)
				out_rgb[2]=max(min(out_rgb[2],255),0)
				#print out_rgb  ############################################## some values are <0
				#pix[x,y]=(int(out_rgb[0]),int(out_rgb[1]),int(out_rgb[2]))
				pix[x,y]=(out_rgb[0],out_rgb[1],out_rgb[2])
				vis[R][G][B]=1
				color_transform[R][G][B]=out_rgb
				#print x,y,pix[x,y] ###################################################
				#if out_rgb[0]<=0 or out_rgb[1]<=0 or out_rgb[2]<=0:
				#	print 'a2',x,y,pix[x,y]

    # 根据当前色的亮度在colors1中的亮度分布，计算在colors2中的对应亮度
	def colorTransformSingleL(self,l):
		for i in range(0,len(self.L1)-1):
			if l>=self.L1[i] and l<=self.L1[i+1]:
				break
		l1=self.L1[i]
		l2=self.L1[i+1]
		s=1 if (l1==l2) else ((l-l1)*1.0/(l2-l1))
		L1=self.L2[i]
		L2=self.L2[i+1]
		L=(L2-L1)*s+L1 
		return int(L)
    # transfor color x according ab1->ab2
	def colorTransformSingleAB(self,ab1,ab2,L,x):
		color1=[L,ab1[0],ab1[1]]
		color2=[L,ab2[0],ab2[1]]
		if self.distance2(color1,color2)<1:
			return color1   # why is color1, not x ???
		d=self.sub(color2,color1)
		x0=self.add(x,d)
		Cb=labIntersect(color1,color2)
		# x-->x0
		if isOutLab(x0):
			#print color2,x0  ###########################
 			xb=labBoundary(color2,x0)
		else:
			xb=labIntersect(x,x0)
		dxx=self.distance2(xb,x)
		dcc=self.distance2(Cb,color1)
		l2=min(1,dxx*1.0/dcc)
		xbn=self.normalize(self.sub(xb,x))
		x_x=math.sqrt(self.distance2(color1,color2)*l2)
		return self.add(x,self.sca_mul(xbn,x_x))

	def omega(self,cs1,Lab,i):
		sum=0
		for j in range(0,len(cs1)):
			#print i,j,self.lambdas[j][i]
			#print 'phi',self.phi(math.sqrt(self.distance2(cs1[j],Lab)))  ################################################################
			#print 'sigma',self.sigma
			#print 'r',math.sqrt(self.distance2(cs1[j],Lab))
			sum+=self.lambdas[j][i]*self.phi(math.sqrt(self.distance2(cs1[j],Lab)))
		#print 'sum',sum #####################################
		return sum

	def getLambda(self,cs1):
		s=[]
		k=len(cs1)
		#print 'cs1' ############################################################################
		#print cs1
		for p in range(0,k):
			tmp=[]
			for q in range(0,k):
				tmp.append(self.phi(math.sqrt(self.distance2(cs1[p],cs1[q]))))
			s.append(tmp)
		matrix=np.mat(s)
		#print 'matrix'  ######################################################################
		#print matrix
		lambdas=np.asarray(matrix.I)  # numpy, to get the inverse of a matrix, remember to use np.asarray to mark matrix.I into arrays !!!
		return lambdas

	def phi(self,r):
		return math.exp(-r*r/(2*self.sigma*self.sigma))

	# sigma应该是标准差吧，但为什么这里是求均值 ？？？
	def getSigma(self,colors):
		sum=0
		l=len(colors)
		for i in range(0,l):
			for j in range(0,l):
				if i==j:
					continue
				sum+=math.sqrt(self.distance2(colors[i],colors[j]))
		return sum/(l*(l-1))

	def lightnessTransformSingle(self,colors1,color_rgb,id):
		Lab=rgb2lab(color_rgb)
		print 'colors2',id,Lab #####################################################
		self.Ls_change[id]=Lab[0]
		for i in range(1,len(self.Ls)):
			self.Ls[i]=self.Ls_change[i]
			self.color_list2[i][0]=self.Ls[i]
		self.color_list2[id]=Lab
		print '1.Ls',self.Ls ###########################################
		# 这里想把colors2的亮度变为递增，原来的算法有问题，如果是id=3， 5 4 3 2 1，则会变为4 3 3 3 3，因此i<id这一部分应该是倒着循环，但是这样的话又有问题了
		#for i in range(1,len(self.Ls)):
		#	if i<id:
		#		if self.Ls[i]>self.Ls[i+1]:
		#			print 'i',i,self.Ls[i],self.Ls[i+1]  ##################################################
		#			self.Ls[i]=self.Ls[i+1]
		#			self.color_list2[i][0]=self.Ls[i+1]
		#	if i>id:
		#		if self.Ls[i]<self.Ls[i-1]:
		#			print 'i',i,self.Ls[i],self.Ls[i-1]  ##################################################
		#			self.Ls[i]=self.Ls[i-1]
		#			self.color_list2[i][0]=self.Ls[i-1]
		for i in range(1,id)[::-1]:
				if self.Ls[i]>self.Ls[i+1]:
					print 'i',i,self.Ls[i],self.Ls[i+1]  ##################################################
					self.Ls[i]=self.Ls[i+1]
					self.color_list2[i][0]=self.Ls[i+1]
		for i in range(id+1,len(self.Ls)):
			if self.Ls[i]<self.Ls[i-1]:
					print 'i',i,self.Ls[i],self.Ls[i-1]  ##################################################
					self.Ls[i]=self.Ls[i-1]
					self.color_list2[i][0]=self.Ls[i-1]
		print '2.Ls',self.Ls ###########################################
	# colors1是聚类出来的颜色，总共有1+K个，第一个是黑色，为了避免聚类出来有很多暗色，不用考虑
	# colors2是输入的新的色板，总共有K个
	def lightnessTransform(self,colors1,colors2):
		for i in range(0,len(colors1)):
			Lab1=rgb2lab(colors1[i])
			self.color_list1[i]=Lab1
			#Lab2=rgb2lab(colors2[i])
			#self.color_list2.append(Lab2)
			self.Ls.append(Lab1[0])
			self.Ls_change.append(Lab1[0])

		for i in range(1,len(colors2)):
			self.lightnessTransformSingle(colors1,colors2[i],i)  # color_list1和color_list2第一个是黑色

	def rendering(self, colors2):
		#self.palette()
		#colors1=self.kmeans()
		colors1=self.centers_rgb
		#self.lightnessTransform(colors1,colors2)
		# 因为colors1和colors2的第0个是忽略的，所以得从第1个开始
		#if(mark==True):
		#	save_theme('res/'+str(id)+'_colors1.png',colors1[1:],400,100)
		#if(mark==True):
		#	save_theme('res/'+str(id)+'_colors2.png',colors2[1:],400,100)
		#len2=len(colors2)
		for i in range(0,min(len(colors1),len(colors2))):
			#self.color_list1[i]=rgb2lab(colors1[i])
			#self.color_list2[i]=rgb2lab(colors2[i])
			self.color_list1.append(rgb2lab(colors1[i]))
			self.color_list2.append(rgb2lab(colors2[i]))
		#self.sort(self.color_list2)  # 先对亮度进行排序
		print 'colors1'
		print colors1
		print 'colors2'
		print colors2
		print 'color_list1'
		print self.color_list1
		print 'color_list2'
		print self.color_list2
		print 'centers_lab'
		print self.centers_lab
		print 'centers_rgb'
		print self.centers_rgb
		for i in range(0,len(self.color_list2)):
			print lab2rgb(self.color_list2[i])
		self.colorTransform(self.color_list1,self.color_list2)
		return self.img_copy
	def showImage(self,img):
		img.show()

	def SingleChangeHue(self,color):
		#if(mark==True):
		#	save_theme('res/'+str(id)+'_colors1.png',self.centers_rgb[1:],400,100)
		#if(mark==True):
			#colors2=[]
			#colors2.append([0,0,0])
			#colors2.append(color)
		#	theme=[]
		#	theme.append(color)
		#	save_theme('res/'+str(id)+'_colors2.png',theme,400,100)
		for i in range(1,len(self.centers_rgb)):
			R,G,B=self.centers_rgb[i]
			h1,s1,l1=hsl.rgb2hsl([R,G,B])
			if s1>0 and l1>0:
				print 'i',i
				break
		#R,G,B=self.centers_rgb[2]  ## main color
		#h1,s1,l1=hsl.RGB_to_HSL(R,G,B)
		h2,s2,l2=hsl.rgb2hsl(color)
		print 'hsl1',h1,s1,l1
		print 'hsl2',h2,s2,l2
		delta1=h2-h1
		delta2=s2/s1
		delta3=l2/l1
		print 'delta',delta1
		pix=self.img_copy.load()
		width=self.img_copy.size[0]
		height=self.img_copy.size[1]
		for x in range(0,width):
			#print x
			for y in range(0,height):
				#print 'y',y
				#print pix[x,y]
				r,g,b=pix[x,y]
				h,s,l = hsl.rgb2hsl([r,g,b])
				h = h+delta1
				if h > 1:
					h = h - 1
				if h < 0:
					h = h + 1
				s = s*delta2
				if s > 1:
					s = 1 
				l = l*delta3
				if l > 1:
					l = 1
				r,g,b = hsl.hsl2rgb([h,s,l])
				pix[x,y]=(r,g,b)
				#if r==255 and g==255 and b==255:
				#	print h,s,l
		return self.img_copy

# 因为聚类是k+1个颜色，所以idx得设为1，传入参数的时候得把第一个排除在外
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

if __name__ == '__main__':
	test_id=22
	filename='input/test3.png'
	colors2=[]
	colors2.append([0,0,0])  ### 第一个忽略
	file = open('colors.txt','r');
	for line in file:
		line=line.strip('\n') # remove '\n' at the end of line
		#a=line.split(" ")[0]
		#b=line.split(" ")[1]
		#c=line.split(" ")[2]
		a,b,c=map(int,line.split())
		colors2.append([a,b,c])
		print [a,b,c]
	file.close();
	l=len(colors2)
	img = Image.open(filename)
	#img.show()
	if img.mode != 'RGB':
		img = img.convert('RGB')
	cluster_k=5
	if l>2:
		pp=Rendering(img,cluster_k)

		start=time.time()
		#print 'start_time',start

		img_rendering=pp.rendering(colors2)

		end=time.time()
		#print 'end_time',end
		print 'Time is %0.2f seconds',(end-start)
		save_theme('res/'+str(test_id)+'_colors1.png',pp.centers_rgb[1:],400,100)
		save_theme('res/'+str(test_id)+'_colors2.png',colors2[1:],400,100)
		img.save('res/'+str(test_id)+'_img.png')
		img_rendering.save('res/'+str(test_id)+'_res.png')
		#img_rendering.show()
	else:
		pp=Rendering(img,cluster_k)
		color=colors2[1]
		print 'center_rgb',pp.centers_rgb
		img_hue=pp.SingleChangeHue(colors2[1])

		
		save_theme('res/'+str(test_id)+'_colors1.png',pp.centers_rgb[1:],400,100)
		theme=[]
		theme.append(color)
		save_theme('res/'+str(test_id)+'_colors2.png',theme,400,100)

		img.save('res/'+str(test_id)+'_img.png')
		img_hue.save('res/'+str(test_id)+'_res.png')
	
	#r1= hsl.HSL_to_RGB(0.5,0,1)
	#r2= hsl.HSL_to_RGB(0.9,0,1)
	#print r1,r2
	#h1,s1,l1=hsl.RGB_to_HSL(230,230,230)
	#print h1,s1,l1


