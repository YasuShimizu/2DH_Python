import numpy as np

nx=5;ny=3
xl=10.;yl=5.

dx=xl/nx;dy=yl/ny
#print(dx,dy)
h=np.zeros([nx+2,ny+2]);u=np.zeros([nx+2,ny+2]);v=np.zeros([nx+2,ny+2])
print(np.shape(h)) # 0～nx+1

hx=np.zeros([nx+1,ny+1])
print(np.shape(hx)) # 0～nx


x = np.linspace(0, xl, nx+1) 
y = np.linspace(0, yl, ny+1)
print(np.shape(x),np.shape(y))

Y,X= np.meshgrid(y,x)
print(np.shape(X),np.shape(Y))

levels=np.arange(0.016, 0.036, 0.002)
print(levels)


