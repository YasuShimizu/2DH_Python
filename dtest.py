import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image, ImageDraw

nx=20;ny=10
xl=40.;yl=10.
xsize=30.
ysize=xsize/xl*yl

dx=xl/nx;dy=yl/ny
#print(dx,dy)
h=np.zeros([nx+1,ny+1]);u=np.zeros([nx+1,ny+1]);v=np.zeros([nx+1,ny+1])
h[:,2]=2
print(np.shape(h))
#print(h)

x = np.linspace(0,xl,nx+1)
y = np.linspace(-yl/2.,yl/2.,ny+1)
#print('x=',x,"\n",'y=',y)

Y,X=np.meshgrid(y,x)
print('X=',np.shape(X))
print('Y=',np.shape(Y))


l=0
lmax=40
r0=yl/2.

while l<lmax:
    xp=l*dx;yp=0.
    print(l,xp)
    for i in np.arange(0,nx+1):
        for j in np.arange(0,ny+1):
            r=np.sqrt((xp-X[i,j])**2+(yp-Y[i,j])**2)
            if r<r0:
                h[i,j]=(r0-r)*3.
                u[i,j]=h[i,j]
            else:
                h[i,j]=0.
                u[i,j]=0.
    fig, ax = plt.subplots(figsize = (xsize, ysize))
    cont=ax.contourf(X, Y, h) 
    vec=ax.quiver(X, Y, u, v)
    xp=[0.,xl/2.,xl/2.,0.,0.];yp=[0.,0.,yl/4.,yl/4.,0.]
    ax.plot(xp,yp)
    ax.fill(xp,yp,color = "coral")
#    im = Image.new('RGB', (500, 250), (128, 128, 128))
#    draw = ImageDraw.Draw(im)
#    draw.polygon(((200, 200), (300, 100), (250, 50)), fill=(255, 255, 0), outline=(0, 0, 0))
    fname="./png/d" +str(l)+ '.png'
    im=plt.savefig(fname)
    l=l+1



