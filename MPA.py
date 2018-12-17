import cv2            #IMPORTING CV2
import numpy as np      #IMPORTING NUMPY
import sys              #IMPORTING SYS MODULE
import math            #IMPORTING MATH MODULE

                                                         #METHOD TO SHOW IMAGES     #TAKING A LIST OF IMAGES AS AN ARGUMENT
def showImages(srcImages):
	# Read images
	for i in range(6):
		img = np.zeros((2000,2000,3), np.uint8)
		img=cv2.imread('m'+str(i)+'.jpg')
		screen_res = 1280, 720
		scale_width = screen_res[0] / img.shape[1]
		scale_height = screen_res[1] / img.shape[0]
		scale = min(scale_width, scale_height)
		window_width = int(img.shape[1] * scale)
		window_height = int(img.shape[0] * scale)
		cv2.imshow(str(i)+" image", srcImages[i])
		cv2.waitKey(0)

def readImages():                                       #FUNCTION TO READ IMAGES FROM THE DIRECTORY
	# Read images
	srcImages=[]                                              
	for i in range(6):
		srcImage = cv2.imread("m"+str(i)+".jpg")
		#cv2.imshow("1",srcImage)
		#cv2.waitKey(0)
		srcImage = np.float32(srcImage)
		srcImages.append(srcImage)
	return srcImages
def conversion(xlist):					#FUNCTION TO CONVERT LIST OF TUPLES INTO NUMPY ARRAY
	dt=np.dtype('int,float')
	np.array(xlist,dtype=dt)
	return xlist

def AbstractingCoordinates(i):                          #FUNCTION TO ABSTRACT THE COORDINATES OF THE POINT CLICKED BY MOUSE
	coordinatesx=[]
	coordinatesy=[]
							#POINT CALLBACK FUNCTION
	def find_point(event,x,y,flags,param):
		if event == cv2.EVENT_LBUTTONDBLCLK:
			coordinatesx.append(x)
			coordinatesy.append(y)
			print('(x,y)== ({},{})'.format(x,y))

							# Create a black image, a window and bind the function to window
	img = np.zeros((2000,2000,3), np.uint8)
	img=cv2.imread('m'+str(i)+'.jpg')
	screen_res = 1280, 720
	scale_width = screen_res[0] / img.shape[1]
	scale_height = screen_res[1] / img.shape[0]
	scale = min(scale_width, scale_height)
	window_width = int(img.shape[1] * scale)
	window_height = int(img.shape[0] * scale)

	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image', window_width, window_height)
	
	cv2.imshow('image', img)
	cv2.setMouseCallback('image',find_point)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	coordinates=(coordinatesx,coordinatesy)
	return coordinates

def writeInTxt(x,y,ii):						#FUNCTION TO WRITE THE ABSTRACTED COORDINATES INTO TXT FILES WHICH WILL BE USED LATER
	f=open(str(ii)+'.txt','a')
	for i in range(4):
		f.write(str(x[i])+ ',' + str(y[i]) + '\n')

def readFromTxt(i):                                            #FUNCTION TO READ THE COORDINATES(X,Y) FROM THE TXT FILES CREATED EARLIER
	# Create an array of intermediatePoints and read intermediatePoints
	intermediatePoints = []
	with open(str(i)+'.txt') as file:
		for line in file:
			x,y = line.split(",")
			intermediatePoints.append((int(x),int(y)))
	return intermediatePoints

q=[]
for i in range(6):                                             #FUNCTION TO ABSTRACT COORDINATES OF ALL THE IMAGES (6 IMAGES)
	x=AbstractingCoordinates(i)
#	print(x)
	writeInTxt(x[0],x[1],i)	
	xx=readFromTxt(i)
	q.append(np.array(conversion(xx)))
	#print(type(q[0]))
	#print(xx)


H, mask = cv2.findHomography(q[0], q[1], cv2.RANSAC, 5.0)         #FIND THE HOMOGRAPHY MATRIX
#print(H)
#print(mask)

inverse = np.linalg.inv(H)	                                #INVERSE OF THE HOMOGRAPHY MATRIX
#print(inverse)	
p=readImages()
d1=p[0].shape
d2=p[1].shape
#print(d1)
#print(d1)
#print(type(d2[0]))
nana1=[[0,0,0],[d1[0],0,0],[d1[0],d1[1],0],[0,d1[1],0]]
nana2=[[0,0,0],[d2[0],0,0],[d2[0],d2[1],0],[0,d2[1],0]]
result = [[0],
         [0],
         [0]]
#print(nana1)
#print(nana2)
res=[]
resx=[]
resy=[]
#pama=nana1[koaml]                                    #TAKING OUT NEW COORDINATES (X,Y) BY MULTIPLYING WITH THE INVERSE HOMOGRAPHY MATRIX
for coordinates in range(4):
	pama=nana2[coordinates]
	#print(pama)
	for i in range(3):
	   # iterate through columns of Y
		for j in range(1):
	       # iterate through rows of Y
			for k in range(3):
				result[i][j] += inverse[i][k] * pama[k]
				#print(k,j)
				#print(nana1[k][j])
	if(result[2][0]!=0):
		for i in range(3):
			for j in range(1):
				result[i][j]=result[i][j]/result[2][0]
	#print(result[0],result[1])
	resx.append(result[0][0])
	resy.append(result[1][0])
	#print(resx)
new_arr=[[0,0],[d2[0],0],[d2[0],d2[1]]]
 
nmaxx=max(0,d2[0])                         
nmaxy=max(0,d2[1])

fmaxx_mod=max(resx)
fmaxy_mod=max(resy)

nminx=min(0,d2[0])
min_new_arr_y=min(0,d2[1])

fminx_mod=min(resx)
fminy_mod=min(resy)

fminx=min(fminx_mod,nmaxx)
fminy=min(fminy_mod,nmaxy)

fmaxx=max(fminx_mod,nmaxx)
fmaxy=max(fminy_mod,nmaxy)

if(fminx<0):
	offset_x=round(abs(fminx))                #ROUNDING OFF THE ABSOLUTE VALUE OF FMINX IF FMINX IS LESS THAN ZERO
else:
	offset_x=0
if(fminy<0):
	offset_y=round(abs(fminy))                 #ROUNDING OFF THE ABSOLUTE VALUE OF FMINY IF FMINY IS LESS THAN ZERO
else:
	offset_y=0

new_x=math.ceil(abs(fmaxx)+abs(fminx))         #TAKING OUT CEIL
new_y=math.ceil(abs(fmaxx)+abs(fminx))
s=[10*new_x,10*new_y]
s_img=np.zeros(s,np.ndarray);
p0=cv2.imread("m"+str(0)+".jpg")                 #READING IMAGE
p1 = cv2.imread("m"+str(1)+".jpg")
d1=p0.shape
d2=p1.shape                                     #TAKING DIMENSIONS
'''
cv2.imshow('1',p0)
cv2.waitKey(0)
cv2.imshow('2',p1)
cv2.waitKey(0) 
'''                                             #FUNCTION TO JOIN TWO IMAGES
#print(type(p0))
for i in range(d1[0]):
	for j in range(d1[1]):
		s_img[int(i+offset_x)][int(j+offset_y)]=p0[i][j]
		
#s_img = cv2.fromarray(s_img)
cv2.imshow("3imges",s_img)
cv2.waitKey(0)

for i in range(new_x):
	for j in range(new_y):
		pama=[i,j,1]
		if(s_img[i][j]==0):
			point = [[0],[0],[0]]
			for i in range(3):
				for j in range(1):
					for k in range(3):
						point[i][j] += H[i][k] * pama[k]

			if(point[2][0]!=0):
				for ii in range(3):
					for jj in range(1):
						point[ii][jj]=point[ii][jj]/point[2][0]
			#point=H*[i, j ,1]
			print(H)
			print(point)
			ct=[]
			ct.append(np.round(point[0][0]))
			ct.append(np.round(point[0][0]))
			if(ct[0]>=1):
				if(ct[0]<=d1[0]):
					if( ct[1]>=1):
						if( ct[1]<=d1[1]):
							v=0
							s_img[i][j]=p0[int(ct[0])][int(ct[1])]
				
cv2.imshow("Merged",s_img)
H, mask = cv2.findHomography(q[0], q[2], cv2.RANSAC, 5.0)         #FIND THE HOMOGRAPHY MATRIX
#print(H)
#print(mask)

inverse = np.linalg.inv(H)	                                #INVERSE OF THE HOMOGRAPHY MATRIX
#print(inverse)	
p=readImages()
d1=p[0].shape
d2=p[2].shape
#print(d1)
#print(d1)
#print(type(d2[0]))
nana1=[[0,0,0],[d1[0],0,0],[d1[0],d1[1],0],[0,d1[1],0]]
nana2=[[0,0,0],[d2[0],0,0],[d2[0],d2[1],0],[0,d2[1],0]]
result = [[0],
         [0],
         [0]]
#print(nana1)
#print(nana2)
res=[]
resx=[]
resy=[]
#pama=nana1[koaml]                                    #TAKING OUT NEW COORDINATES (X,Y) BY MULTIPLYING WITH THE INVERSE HOMOGRAPHY MATRIX
for coordinates in range(4):
	pama=nana2[coordinates]
	#print(pama)
	for i in range(3):
	   # iterate through columns of Y
		for j in range(1):
	       # iterate through rows of Y
			for k in range(3):
				result[i][j] += inverse[i][k] * pama[k]
				#print(k,j)
				#print(nana1[k][j])
	if(result[2][0]!=0):
		for i in range(3):
			for j in range(1):
				result[i][j]=result[i][j]/result[2][0]
	#print(result[0],result[1])
	resx.append(result[0][0])
	resy.append(result[1][0])
	#print(resx)
new_arr=[[0,0],[d2[0],0],[d2[0],d2[1]]]
 
nmaxx=max(0,d2[0])                         
nmaxy=max(0,d2[1])

fmaxx_mod=max(resx)
fmaxy_mod=max(resy)

nminx=min(0,d2[0])
min_new_arr_y=min(0,d2[1])

fminx_mod=min(resx)
fminy_mod=min(resy)

fminx=min(fminx_mod,nmaxx)
fminy=min(fminy_mod,nmaxy)

fmaxx=max(fminx_mod,nmaxx)
fmaxy=max(fminy_mod,nmaxy)

if(fminx<0):
	offset_x=round(abs(fminx))                #ROUNDING OFF THE ABSOLUTE VALUE OF FMINX IF FMINX IS LESS THAN ZERO
else:
	offset_x=0
if(fminy<0):
	offset_y=round(abs(fminy))                 #ROUNDING OFF THE ABSOLUTE VALUE OF FMINY IF FMINY IS LESS THAN ZERO
else:
	offset_y=0

new_x=math.ceil(abs(fmaxx)+abs(fminx))         #TAKING OUT CEIL
new_y=math.ceil(abs(fmaxx)+abs(fminx))
s=[10*new_x,10*new_y]
s_img=np.zeros(s,np.ndarray);
p0=cv2.imread("m"+str(0)+".jpg")                 #READING IMAGE
p1 = cv2.imread("m"+str(2)+".jpg")
d1=p0.shape
d2=p1.shape                                     #TAKING DIMENSIONS
'''
cv2.imshow('1',p0)
cv2.waitKey(0)
cv2.imshow('2',p1)
cv2.waitKey(0) 
'''                                             #FUNCTION TO JOIN TWO IMAGES
#print(type(p0))
for i in range(d1[0]):
	for j in range(d1[1]):
		s_img[int(i+offset_x)][int(j+offset_y)]=p0[i][j]
		
#s_img = cv2.fromarray(s_img)
cv2.imshow("3imges",s_img)
cv2.waitKey(0)

for i in range(new_x):
	for j in range(new_y):
		pama=[i,j,1]
		if(s_img[i][j]==0):
			point = [[0],[0],[0]]
			for i in range(3):
				for j in range(1):
					for k in range(3):
						point[i][j] += H[i][k] * pama[k]

			if(point[2][0]!=0):
				for ii in range(3):
					for jj in range(1):
						point[ii][jj]=point[ii][jj]/point[2][0]
			#point=H*[i, j ,1]
			print(H)
			print(point)
			ct=[]
			ct.append(np.round(point[0][0]))
			ct.append(np.round(point[0][0]))
			if(ct[0]>=1):
				if(ct[0]<=d1[0]):
					if( ct[1]>=1):
						if( ct[1]<=d1[1]):
							v=0
							s_img[i][j]=p0[int(ct[0])][int(ct[1])]
				
cv2.imshow("Merged",s_img)
H, mask = cv2.findHomography(q[0], q[3], cv2.RANSAC, 5.0)         #FIND THE HOMOGRAPHY MATRIX
#print(H)
#print(mask)

inverse = np.linalg.inv(H)	                                #INVERSE OF THE HOMOGRAPHY MATRIX
#print(inverse)	
p=readImages()
d1=p[0].shape
d2=p[3].shape
#print(d1)
#print(d1)
#print(type(d2[0]))
nana1=[[0,0,0],[d1[0],0,0],[d1[0],d1[1],0],[0,d1[1],0]]
nana2=[[0,0,0],[d2[0],0,0],[d2[0],d2[1],0],[0,d2[1],0]]
result = [[0],
         [0],
         [0]]
#print(nana1)
#print(nana2)
res=[]
resx=[]
resy=[]
#pama=nana1[koaml]                                    #TAKING OUT NEW COORDINATES (X,Y) BY MULTIPLYING WITH THE INVERSE HOMOGRAPHY MATRIX
for coordinates in range(4):
	pama=nana2[coordinates]
	#print(pama)
	for i in range(3):
	   # iterate through columns of Y
		for j in range(1):
	       # iterate through rows of Y
			for k in range(3):
				result[i][j] += inverse[i][k] * pama[k]
				#print(k,j)
				#print(nana1[k][j])
	if(result[2][0]!=0):
		for i in range(3):
			for j in range(1):
				result[i][j]=result[i][j]/result[2][0]
	#print(result[0],result[1])
	resx.append(result[0][0])
	resy.append(result[1][0])
	#print(resx)
new_arr=[[0,0],[d2[0],0],[d2[0],d2[1]]]
 
nmaxx=max(0,d2[0])                         
nmaxy=max(0,d2[1])

fmaxx_mod=max(resx)
fmaxy_mod=max(resy)

nminx=min(0,d2[0])
min_new_arr_y=min(0,d2[1])

fminx_mod=min(resx)
fminy_mod=min(resy)

fminx=min(fminx_mod,nmaxx)
fminy=min(fminy_mod,nmaxy)

fmaxx=max(fminx_mod,nmaxx)
fmaxy=max(fminy_mod,nmaxy)

if(fminx<0):
	offset_x=round(abs(fminx))                #ROUNDING OFF THE ABSOLUTE VALUE OF FMINX IF FMINX IS LESS THAN ZERO
else:
	offset_x=0
if(fminy<0):
	offset_y=round(abs(fminy))                 #ROUNDING OFF THE ABSOLUTE VALUE OF FMINY IF FMINY IS LESS THAN ZERO
else:
	offset_y=0

new_x=math.ceil(abs(fmaxx)+abs(fminx))         #TAKING OUT CEIL
new_y=math.ceil(abs(fmaxx)+abs(fminx))
s=[10*new_x,10*new_y]
s_img=np.zeros(s,np.ndarray);
p0=cv2.imread("m"+str(0)+".jpg")                 #READING IMAGE
p1 = cv2.imread("m"+str(3)+".jpg")
d1=p0.shape
d2=p1.shape                                     #TAKING DIMENSIONS
'''
cv2.imshow('1',p0)
cv2.waitKey(0)
cv2.imshow('2',p1)
cv2.waitKey(0) 
'''                                             #FUNCTION TO JOIN TWO IMAGES
#print(type(p0))
for i in range(d1[0]):
	for j in range(d1[1]):
		s_img[int(i+offset_x)][int(j+offset_y)]=p0[i][j]
		
#s_img = cv2.fromarray(s_img)
cv2.imshow("3imges",s_img)
cv2.waitKey(0)

for i in range(new_x):
	for j in range(new_y):
		pama=[i,j,1]
		if(s_img[i][j]==0):
			point = [[0],[0],[0]]
			for i in range(3):
				for j in range(1):
					for k in range(3):
						point[i][j] += H[i][k] * pama[k]

			if(point[2][0]!=0):
				for ii in range(3):
					for jj in range(1):
						point[ii][jj]=point[ii][jj]/point[2][0]
			#point=H*[i, j ,1]
			print(H)
			print(point)
			ct=[]
			ct.append(np.round(point[0][0]))
			ct.append(np.round(point[0][0]))
			if(ct[0]>=1):
				if(ct[0]<=d1[0]):
					if( ct[1]>=1):
						if( ct[1]<=d1[1]):
							v=0
							s_img[i][j]=p0[int(ct[0])][int(ct[1])]
				
cv2.imshow("Merged3",s_img)

H, mask = cv2.findHomography(q[0], q[4], cv2.RANSAC, 5.0)         #FIND THE HOMOGRAPHY MATRIX
#print(H)
#print(mask)

inverse = np.linalg.inv(H)	                                #INVERSE OF THE HOMOGRAPHY MATRIX
#print(inverse)	
p=readImages()
d1=p[0].shape
d2=p[4].shape
#print(d1)
#print(d1)
#print(type(d2[0]))
nana1=[[0,0,0],[d1[0],0,0],[d1[0],d1[1],0],[0,d1[1],0]]
nana2=[[0,0,0],[d2[0],0,0],[d2[0],d2[1],0],[0,d2[1],0]]
result = [[0],
         [0],
         [0]]
#print(nana1)
#print(nana2)
res=[]
resx=[]
resy=[]
#pama=nana1[koaml]                                    #TAKING OUT NEW COORDINATES (X,Y) BY MULTIPLYING WITH THE INVERSE HOMOGRAPHY MATRIX
for coordinates in range(4):
	pama=nana2[coordinates]
	#print(pama)
	for i in range(3):
	   # iterate through columns of Y
		for j in range(1):
	       # iterate through rows of Y
			for k in range(3):
				result[i][j] += inverse[i][k] * pama[k]
				#print(k,j)
				#print(nana1[k][j])
	if(result[2][0]!=0):
		for i in range(3):
			for j in range(1):
				result[i][j]=result[i][j]/result[2][0]
	#print(result[0],result[1])
	resx.append(result[0][0])
	resy.append(result[1][0])
	#print(resx)
new_arr=[[0,0],[d2[0],0],[d2[0],d2[1]]]
 
nmaxx=max(0,d2[0])                         
nmaxy=max(0,d2[1])

fmaxx_mod=max(resx)
fmaxy_mod=max(resy)

nminx=min(0,d2[0])
min_new_arr_y=min(0,d2[1])

fminx_mod=min(resx)
fminy_mod=min(resy)

fminx=min(fminx_mod,nmaxx)
fminy=min(fminy_mod,nmaxy)

fmaxx=max(fminx_mod,nmaxx)
fmaxy=max(fminy_mod,nmaxy)

if(fminx<0):
	offset_x=round(abs(fminx))                #ROUNDING OFF THE ABSOLUTE VALUE OF FMINX IF FMINX IS LESS THAN ZERO
else:
	offset_x=0
if(fminy<0):
	offset_y=round(abs(fminy))                 #ROUNDING OFF THE ABSOLUTE VALUE OF FMINY IF FMINY IS LESS THAN ZERO
else:
	offset_y=0

new_x=math.ceil(abs(fmaxx)+abs(fminx))         #TAKING OUT CEIL
new_y=math.ceil(abs(fmaxx)+abs(fminx))
s=[10*new_x,10*new_y]
s_img=np.zeros(s,np.ndarray);
p0=cv2.imread("m"+str(0)+".jpg")                 #READING IMAGE
p1 = cv2.imread("m"+str(4)+".jpg")
d1=p0.shape
d2=p1.shape                                     #TAKING DIMENSIONS
'''
cv2.imshow('1',p0)
cv2.waitKey(0)
cv2.imshow('2',p1)
cv2.waitKey(0) 
'''                                             #FUNCTION TO JOIN TWO IMAGES
#print(type(p0))
for i in range(d1[0]):
	for j in range(d1[1]):
		s_img[int(i+offset_x)][int(j+offset_y)]=p0[i][j]
		
#s_img = cv2.fromarray(s_img)
cv2.imshow("3imges",s_img)
cv2.waitKey(0)

for i in range(new_x):
	for j in range(new_y):
		pama=[i,j,1]
		if(s_img[i][j]==0):
			point = [[0],[0],[0]]
			for i in range(3):
				for j in range(1):
					for k in range(3):
						point[i][j] += H[i][k] * pama[k]

			if(point[2][0]!=0):
				for ii in range(3):
					for jj in range(1):
						point[ii][jj]=point[ii][jj]/point[2][0]
			#point=H*[i, j ,1]
			print(H)
			print(point)
			ct=[]
			ct.append(np.round(point[0][0]))
			ct.append(np.round(point[0][0]))
			if(ct[0]>=1):
				if(ct[0]<=d1[0]):
					if( ct[1]>=1):
						if( ct[1]<=d1[1]):
							v=0
							s_img[i][j]=p0[int(ct[0])][int(ct[1])]
				
cv2.imshow("Merged4",s_img)
H, mask = cv2.findHomography(q[0], q[5], cv2.RANSAC, 5.0)         #FIND THE HOMOGRAPHY MATRIX
#print(H)
#print(mask)

inverse = np.linalg.inv(H)	                                #INVERSE OF THE HOMOGRAPHY MATRIX
#print(inverse)	
p=readImages()
d1=p[0].shape
d2=p[1].shape
#print(d1)
#print(d1)
#print(type(d2[0]))
nana1=[[0,0,0],[d1[0],0,0],[d1[0],d1[1],0],[0,d1[1],0]]
nana2=[[0,0,0],[d2[0],0,0],[d2[0],d2[1],0],[0,d2[1],0]]
result = [[0],
         [0],
         [0]]
#print(nana1)
#print(nana2)
res=[]
resx=[]
resy=[]
#pama=nana1[koaml]                                    #TAKING OUT NEW COORDINATES (X,Y) BY MULTIPLYING WITH THE INVERSE HOMOGRAPHY MATRIX
for coordinates in range(4):
	pama=nana2[coordinates]
	#print(pama)
	for i in range(3):
	   # iterate through columns of Y
		for j in range(1):
	       # iterate through rows of Y
			for k in range(3):
				result[i][j] += inverse[i][k] * pama[k]
				#print(k,j)
				#print(nana1[k][j])
	if(result[2][0]!=0):
		for i in range(3):
			for j in range(1):
				result[i][j]=result[i][j]/result[2][0]
	#print(result[0],result[1])
	resx.append(result[0][0])
	resy.append(result[1][0])
	#print(resx)
new_arr=[[0,0],[d2[0],0],[d2[0],d2[1]]]
 
nmaxx=max(0,d2[0])                         
nmaxy=max(0,d2[1])

fmaxx_mod=max(resx)
fmaxy_mod=max(resy)

nminx=min(0,d2[0])
min_new_arr_y=min(0,d2[1])

fminx_mod=min(resx)
fminy_mod=min(resy)

fminx=min(fminx_mod,nmaxx)
fminy=min(fminy_mod,nmaxy)

fmaxx=max(fminx_mod,nmaxx)
fmaxy=max(fminy_mod,nmaxy)

if(fminx<0):
	offset_x=round(abs(fminx))                #ROUNDING OFF THE ABSOLUTE VALUE OF FMINX IF FMINX IS LESS THAN ZERO
else:
	offset_x=0
if(fminy<0):
	offset_y=round(abs(fminy))                 #ROUNDING OFF THE ABSOLUTE VALUE OF FMINY IF FMINY IS LESS THAN ZERO
else:
	offset_y=0

new_x=math.ceil(abs(fmaxx)+abs(fminx))         #TAKING OUT CEIL
new_y=math.ceil(abs(fmaxx)+abs(fminx))
s=[10*new_x,10*new_y]
s_img=np.zeros(s,np.ndarray);
p0=cv2.imread("m"+str(0)+".jpg")                 #READING IMAGE
p1 = cv2.imread("m"+str(5)+".jpg")
d1=p0.shape
d2=p1.shape                                     #TAKING DIMENSIONS
'''
cv2.imshow('1',p0)
cv2.waitKey(0)
cv2.imshow('2',p1)
cv2.waitKey(0) 
'''                                             #FUNCTION TO JOIN TWO IMAGES
#print(type(p0))
for i in range(d1[0]):
	for j in range(d1[1]):
		s_img[int(i+offset_x)][int(j+offset_y)]=p0[i][j]
		
#s_img = cv2.fromarray(s_img)
cv2.imshow("3imges",s_img)
cv2.waitKey(0)

for i in range(new_x):
	for j in range(new_y):
		pama=[i,j,1]
		if(s_img[i][j]==0):
			point = [[0],[0],[0]]
			for i in range(3):
				for j in range(1):
					for k in range(3):
						point[i][j] += H[i][k] * pama[k]

			if(point[2][0]!=0):
				for ii in range(3):
					for jj in range(1):
						point[ii][jj]=point[ii][jj]/point[2][0]
			#point=H*[i, j ,1]
			print(H)
			print(point)
			ct=[]
			ct.append(np.round(point[0][0]))
			ct.append(np.round(point[0][0]))
			if(ct[0]>=1):
				if(ct[0]<=d1[0]):
					if( ct[1]>=1):
						if( ct[1]<=d1[1]):
							v=0
							s_img[i][j]=p0[int(ct[0])][int(ct[1])]
				
cv2.imshow("Merged5",s_img)



