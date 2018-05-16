# -*- coding: utf-8 -*-
import math 
# 参考了matlab 和谐度评估中rgb和lab之间的转化函数，原先的可能有点误差
def rgb2lab(inputColor):
	RGB=[0,0,0]
	for i in range(0,len(inputColor)):
		RGB[i]=inputColor[i]/255.0
		#if v>0.04045:
		#	v=pow((v+0.055)/1.0,2.4)
		#else:
		#	v/=12.92
		#RGB[i]=100*v
	X=RGB[0]*0.4124+RGB[1]*0.3576+RGB[2]*0.1805
	Y=RGB[0]*0.2126+RGB[1]*0.7152+RGB[2]*0.0722
	Z=RGB[0]*0.0193+RGB[1]*0.1192+RGB[2]*0.9505
	XYZ=[X,Y,Z]
	XYZ[0]/=95.045/100
	XYZ[1]/=100.0/100
	XYZ[2]/=108.875/100
	#Y3=math.pow(XYZ[1],1.0/3)
	L=0
	for i in range(0,3):
		v=XYZ[i]
		if v>0.008856:
			v=pow(v,1.0/3)
			if i==1:
				L=116.0*v-16.0
		else:
			v*=7.787
			v+=16.0/116
			if i==1:
				L=903.3*XYZ[i]
		XYZ[i]=v
	#L=116.0*XYZ[1]-16
	a=500.0*(XYZ[0]-XYZ[1])
	b=200.0*(XYZ[1]-XYZ[2])
	Lab=[int(L),int(a),int(b)]
	return Lab

def lab2rgb(inputColor):
	L=inputColor[0]
	a=inputColor[1]
	b=inputColor[2]
	#d=6.0/29
	T1=0.008856
	T2=0.206893
	d=T2
	fy =math.pow( (L + 16) / 116.0,3)
	fx = fy + a / 500.0
	fz = fy - b / 200.0
	#Y = fy > d ? fy * fy * fy : (fy - 16.0 / 116) * 3 * d * d
	fy = (fy) if (fy > T1) else ( L/903.3)
	Y=fy
	fy=(math.pow(fy,1.0/3)) if (fy > T1) else (7.787*fy+16.0/116)  # calculate XYZ[1], XYZ[0]=a/500.0+XYZ[1]

	# compute original XYZ[0]
	fx=fy+a/500.0
	X=(math.pow(fx,3.0)) if (fx > T2) else ((fx-16.0/116)/7.787)  # v^3>T1, so v>T1^(1/3)=

	# compute original XYZ[2]
	fz=fy-b/200.0
	Z=(math.pow(fz,3.0)) if (fz >T2) else ((fz-16.0/116)/7.787)

	X*=0.95045
	Z*=1.08875
	R = 3.240479 * X + (-1.537150) * Y + (-0.498535) * Z
	G = (-0.969256) * X + 1.875992 * Y + 0.041556 * Z
	B = 0.055648 * X + (-0.204043) * Y + 1.057311 * Z
	#R = max(min(R,1),0)
	#G = max(min(G,1),0)
	#B = max(min(B,1),0)
	RGB = [R, G, B];
	#console.log(RGB);
	for i in range(0,3):
		RGB[i] = int(round(RGB[i] * 255))
	return RGB

def isOutRGB(RGB):
	for i in range(0,3):
		if RGB[i]<0 or RGB[i]>255:
			return True
	return False

def isOutLab(Lab):
	return isOutRGB(lab2rgb(Lab))
	#if Lab[0] <0 or Lab[0]>100.0:
	#	return True
	#for i in range(1,3):
	#	if Lab[i]<=-128.0 or Lab[0]>=127.0:
	#		return True
	#return False

def isEqual(c1, c2):
	for i in range(0,len(c1)):
		if c1[i]!=c2[i]:
			return False
	return True
# 当pout在边界外的时候，二分查找与边界的交点
def labBoundary(pin, pout):
	mid = [];
	for i in range(0,len(pin)):
		mid.append((pin[i]+pout[i])/2.0)
	RGBin = lab2rgb(pin);
	RGBout = lab2rgb(pout);
	RGBmid = lab2rgb(mid);
	#print 'Lab',pin,mid,pout
	#print 'RGB',RGBin,RGBmid,RGBout ######################################
	#print distance2(pin,pout)
	if (distance2(pin,pout)<1 or isEqual(RGBin, RGBout)):
		return mid
	if isOutRGB(RGBmid):
		return labBoundary(pin, mid)
	else:
		return labBoundary(mid, pout)
# 这里应该是寻找p1 p2延长线与边界的交点
def labIntersect(p1, p2):
	if isOutLab(p2):
		return labBoundary(p1,p2)
	else:
		return labIntersect(p2,add(p2,sub(p2,p1)))
		#return labIntersect(p1,add(p1,sca_mul(sub(p2,p1),10)))
def add(c1,c2):
	res=[]
	for i in range(0,len(c1)):
		res.append(c1[i]+c2[i])
	return res

def sub(c1,c2):
	res=[]
	for i in range(0,len(c1)):
		res.append(c1[i]-c2[i])
	return res
def distance2(c1,c2):
	res=0;
	for i in range(0,len(c1)):
		res+=(c1[i]-c2[i])*(c1[i]-c2[i])
	return res
def sca_mul(c,k):
		res=[]
		for i in range(0,len(c)):
			res.append(c[i]*k)
		return res