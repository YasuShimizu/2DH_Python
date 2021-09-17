import numpy as np
import math, copy, os, yaml, subprocess
import initial,obstacle,boundary,stgg,cfxy,rhs,diffusion
import newgrd,cip2d,uxvx,fric,mkzero,hcal

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib._version import get_versions as mplv

# Open Config File
with open('config.yml','r', encoding='utf-8') as yml:   
    config = yaml.load(yml)

snu_0=float(config['snu_0'])
hmin=float(config['hmin'])
cw=float(config['cw'])
ep_alpha=float(config['ep_alpha'])

nx=int(config['nx']);ny=int(config['ny'])
chl=float(config['chl']);chb=float(config['chb'])
slope=float(config['slope']);xsize=float(config['xsize'])

qp=float(config['qp'])
snm=float(config['snm'])
g=float(config['g'])

j_west=int(config['j_west']);j_east=int(config['j_east'])
j_hdown=int(config['j_hdown'])

alh=float(config['alh']); lmax=int(config['lmax'])
etime=float(config['etime']);tuk=float(config['tuk'])
dt=float(config['dt'])

nx1=nx+1; ny1=ny+1;nym=int(ny/2)
nx2=nx+2; ny2=ny+2

xl=chl;yl=chb
ysize=xsize/xl*yl
dx=chl/nx; dy=chb/ny; area=dx*dy
g_sqrt=np.sqrt(g)
errmax=hmin
it_out=int(tuk/dt)

prm=np.zeros([nx2,ny2])
u=np.zeros([nx2,ny2]); un=np.zeros([nx2,ny2]); v=np.zeros([nx2,ny2]); vn=np.zeros([nx2,ny2])
hs=np.zeros([nx2,ny2]);  h=np.zeros([nx2,ny2]); hn=np.zeros([nx2,ny2])
ijh=np.zeros([nx2,ny2],dtype=int)
v_up=np.zeros([nx2,ny2]); hs_up=np.zeros([nx2,ny2]); u_vp=np.zeros([nx2,ny2]); hs_vp=np.zeros([nx2,ny2])
eta=np.zeros([nx2,ny2]); ep=np.zeros([nx2,ny2]); ep_x=np.zeros([nx2,ny2]); usta=np.zeros([nx2,ny2])
up=np.zeros([nx2,ny2]); vp=np.zeros([nx2,ny2])
qu=np.zeros([nx2,ny2]); qv=np.zeros([nx2,ny2]); qc=np.zeros([nx2])
x_center=np.zeros([nx2]); eta_center=np.zeros([nx2]); h_center=np.zeros([nx2])
qu_center=np.zeros([nx2]);hs_center=np.zeros([nx2])
gux=np.zeros([nx2,ny2]); guy=np.zeros([nx2,ny2]); gvx=np.zeros([nx2,ny2]); gvy=np.zeros([nx2,ny2])
gux_n=np.zeros([nx2,ny2]); guy_n=np.zeros([nx2,ny2])
gvx_n=np.zeros([nx2,ny2]); gvy_n=np.zeros([nx2,ny2])
cfx=np.zeros([nx2,ny2]); cfy=np.zeros([nx2,ny2]); qbx=np.zeros([nx2,ny2]); qby=np.zeros([nx2,ny2])
uvis=np.zeros([nx2,ny2]);uvis_x=np.zeros([nx2,ny2]);uvis_y=np.zeros([nx2,ny2])
vvis=np.zeros([nx2,ny2]);vvis_x=np.zeros([nx2,ny2]);vvis_y=np.zeros([nx2,ny2])
fn=np.zeros([nx2,ny2]);gxn=np.zeros([nx2,ny2]);gyn=np.zeros([nx2,ny2])
ux=np.zeros([nx1,ny1]);vx=np.zeros([nx1,ny1]); uv2=np.zeros([nx1,ny1])
hx=np.zeros([nx1,ny1]);hsx=np.zeros([nx1,ny1]);vor=np.zeros([nx1,ny1])

xf=np.zeros([5]);yf=np.zeros([5])

eta=initial.eta_init(eta,nx,ny,slope,dx,chl)  #Initial Bed Elevation
x_center,eta_center=initial.xe_center(x_center,eta_center,nx,slope,dx,chl)

x = np.linspace(0, chl, nx1)
y = np.linspace(0, chb, ny1)

Y,X= np.meshgrid(y,x)

#print(nx,ny)
#print(np.shape(X),np.shape(hx))
#exit()

#Basic Hydraulic Values
hs0=(snm*qp/chb/math.sqrt(slope))**(3/5)
u0=qp/(hs0*chb)
qu0=u0*hs0*dy
hlstep=0.002
hlmin=int(hs0*0.2/hlstep)*hlstep
hlmax=int(hs0*2./hlstep)*hlstep
levels=np.arange(hlmin,hlmax,hlstep)
#print(hlmin,hlmax,hlstep)

ulstep=0.05
ulmin=int(u0*0./ulstep)*ulstep
ulmax=int(u0*4./ulstep)*ulstep
ulevels=np.arange(ulmin,ulmax,ulstep)

vlstep=0.5
vlmax=8.; vlmin=-vlmax
vlevels=np.arange(vlmin,vlmax,vlstep)

#print(vlevels, len(vlevels))
m=len(vlevels)
#print(vlevels[0],vlevels[m-1])
vlevels[0]=vlevels[0]-2.
vlevels[m-1]=vlevels[m-1]+2.
#print(vlevels)


#Downstream Uniform Flow Depth
if j_east==0 and j_hdown==1:
    h_down=eta_center[nx+1]+hs0

#fig=plt.figure()
#im=plt.title("Longitudinal Bed Profile")
#im=plt.xlabel("x(m)"); im=plt.ylabel("Elevation(m)")
#im=plt.plot(x_center,eta_center,'r')
#plt.show()

u=initial.u_init(u,u0,nx,ny); un=u            #Initial Velocities
h,hs=initial.h_init(h,hs,eta,hs0,nx,ny); hn=h #Initial Depth and Water Surface Elevation
ep,ep_x=initial.ep_init(ep,ep_x,nx,ny,snu_0)

ijh=obstacle.ob_ini(ijh,nx,ny)  # Setup Obstacle Cells

#print(ijh)
#for i in np.arange(0,nx+1):
#    print(eta[i,nym],hs[i,nym],ep[i,nym])

h,hs=boundary.h_bound(h,hs,eta,nx,ny,j_west,j_east,j_hdown,h_down)
h_center[:]=h[:,nym]

hn=copy.copy(h) 

#im=plt.title("Longitudinal Bed Profile")
#im=plt.xlabel("x(m)"); im=plt.ylabel("Elevation(m)")
#im=plt.plot(x_center,eta_center,'r')
#im=plt.plot(x_center,h_center,'b')
#plt.show()


u=boundary.u_bound(u,nx,ny,j_west,j_east,ijh,u0); un=u
v=boundary.v_bound(v,nx,ny,ijh)       ; vn=v

hs_up=stgg.hs_up_c(hs_up,hs,nx,ny)
hs_vp=stgg.hs_vp_c(hs_vp,hs,nx,ny)

qu,qc=rhs.qu_cal(qu,qc,u,nx,ny,dy,hs_up)
qv=rhs.qv_cal(qv,v,nx,ny,dx,hs_vp)

#print('qc=',qc)
qadj=qc[0]/qp
u_input=u0/qadj

#gux,guy,gvx,gvy=initial.diffs_init(gux,guy,gvx,gvy,u,v,nx,ny,dx,dy)
#gux,guy=boundary.gbound_u(gux,guy,ijh,nx,ny)
#gvx,gvy=boundary.gbound_v(gvx,gvy,ijh,nx,ny)
#print(gux)

u_vp=stgg.u_vp_c(u_vp,u,nx,ny)
hs_vp=stgg.hs_vp_c(hs_vp,hs,nx,ny)

v_up=stgg.v_up_c(v_up,v,nx,ny)
hs_up=stgg.hs_up_c(hs_up,hs,nx,ny)
#print(hs_up)

time=0.
icount=0

#fig=plt.figure()
#ims=[]

#fig = plt.figure(figsize=(11, 7), dpi=100)

#fig=plt.figure()
#fig, ax = plt.subplots()

#im=plt.title("Velocity Vectors")
#im=plt.xlabel("x")
#im=plt.ylabel("y")

#ux,vx,hx,uv2=uxvx.uv(ux,vx,uv2,hx,u,v,h,nx,ny)

#print(np.round(ux,5))

#im=plt.quiver(X, Y, ux, vx)
#ims.append([im])
#plt.show()

nfile=0
os.system("del /Q .\png\*.png")
iskip=1
l=0

########### Main ############

while time<= etime:
    usta,ep,ep_x=fric.us_cal(usta,ep,ep_x,u,v,hs,nx,ny,snm,g_sqrt,hmin,ep_alpha)
    if icount%it_out==0:
        print('time=',np.round(time,3),l)
#        print(np.round(qc[0:20],3))
#        print('qc=',qc)
        nfile=nfile+1
#        print(np.round(vn,5))
        ux,vx,uv2,hx,hsx=uxvx.uv(ux,vx,uv2,hx,hsx,u,v,h,hs,nx,ny)
        vor=uxvx.vortex(vor,ux,vx,nx,ny,dx,dy)
#        im=plt.contourf(X, Y, hx)  
#        im=plt.colorbar()
#        ims.append(im)
        
#        print('X=',np.round(X,5),'Y=',np.round(Y,5))
#        print('ux=',np.round(ux,5),'vx=',np.round(ux,5))
    

        fig, ax = plt.subplots(figsize = (xsize, ysize))
        
#        cont=ax.contourf(X, Y, hsx,levels) 

#        cont=ax.contourf(X, Y, uv2,ulevels) 
        cont=ax.contourf(X, Y, vor, vlevels, cmap='coolwarm') 

#        cont=ax.contourf(X, Y, uv2) 
        vect=ax.quiver(X[::iskip], Y[::iskip], ux[::iskip], vx[::iskip], \
            width=0.002,headwidth=3)
        fig.colorbar(cont)
        for i in np.arange(1,nx+1):
            for j in np.arange(1,ny+1):
                if ijh[i,j]>=0.1:
                    xf[0]=x[i-1];yf[0]=y[j-1]
                    xf[1]=x[i]  ;yf[1]=y[j-1]
                    xf[2]=x[i]  ;yf[2]=y[j]
                    xf[3]=x[i-1];yf[3]=y[j]
                    xf[4]=x[i-1];yf[4]=y[j-1]
                    ax.fill(xf,yf,color = "green")
        fname="./png/" + 'f%04d' % nfile + '.png'
#        print(fname)
        im=plt.savefig(fname)
        plt.clf()
        plt.close()


#1d        h_center[:]=h[:,nym];hs_center[:]=hs[:,nym]
#1d        qu_center[:]=qu[:,nym]
#1d        im=plt.title("Longitudinal Qu")
#1d        im=plt.xlabel("x(m)"); im=plt.ylabel("Qu")
#1d        im=plt.plot(x_center,eta_center,'r')
#1d        im=plt.plot(x_center,qu_center,'b')
#1d        im=plt.plot(x_center,hs_center,'g')
#1d        ims.append(im)

#Velocities in Non Advection Phase
    l=0
    while l<lmax:
        v_up=stgg.v_up_c(v_up,vn,nx,ny)
        hs_up=stgg.hs_up_c(hs_up,hs,nx,ny)
        cfx=cfxy.cfxc(cfx,nx,ny,hs,un,g,snm,v_up,hs_up)
        un=rhs.un_cal(un,u,nx,ny,dx,cfx,hn,g,dt)
        un=boundary.u_bound(un,nx,ny,j_west,j_east,ijh,u_input)
        qu,qc=rhs.qu_cal(qu,qc,un,nx,ny,dy,hs_up)

        u_vp=stgg.u_vp_c(u_vp,un,nx,ny)
        hs_vp=stgg.hs_vp_c(hs_vp,hs,nx,ny)
        cfy=cfxy.cfyc(cfy,nx,ny,hs,vn,g,snm,u_vp,hs_vp)
        vn=rhs.vn_cal(vn,v,nx,ny,dy,cfy,hn,g,dt)
        vn=boundary.v_bound(vn,nx,ny,ijh)
        qv=rhs.qv_cal(qv,vn,nx,ny,dx,hs_vp)

        hn,hs,err=hcal.hh(hn,h,hs,eta,qu,qv,ijh,area,alh,hmin,nx,ny,dt)
        hn,hs=boundary.h_bound(hn,hs,eta,nx,ny,j_west,j_east,j_hdown,h_down)
#        for i in np.arange(1,40):
#            for j in np.arange(10,11):
#                print(i,j, np.round(hn[i,j]-h[i,j],8))

#        print('l,err=',l,err)
        if err<errmax:
            break
        l=l+1

#Diffusion    
    un=diffusion.diff_u(un,uvis,uvis_x,uvis_y,nx,ny,dx,dy,dt,ep,ep_x,cw)
    un=boundary.u_bound(un,nx,ny,j_west,j_east,ijh,u_input)
    vn=diffusion.diff_v(vn,vvis,vvis_x,vvis_y,nx,ny,dx,dy,dt,ep,ep_x)
    vn=boundary.v_bound(vn,nx,ny,ijh)

#Differentials in Non Advection Phase
    gux,guy=newgrd.ng_u(gux,guy,u,un,nx,ny,dx,dy)
    gux,guy=boundary.gbound_u(gux,guy,ijh,nx,ny)
    gvx,gvy=newgrd.ng_v(gvx,gvy,v,vn,nx,ny,dx,dy)
    gvx,gvy=boundary.gbound_v(gvx,gvy,ijh,nx,ny)

#Advection Phase
    fn,gxn,gyn=mkzero.z0(fn,gxn,gyn,nx,ny)
    v_up=stgg.v_up_c(v_up,v,nx,ny) 
    fn,gxn,gyn=cip2d.u_cal1(un,gux,guy,u,v_up,fn,gxn,gyn,nx,ny,dx,dy,dt)
    un,gux,guy=cip2d.u_cal2(fn,gxn,gyn,u,v_up,un,gux,guy,nx,ny,dx,dy,dt)
    un=boundary.u_bound(un,nx,ny,j_west,j_east,ijh,u_input)
    gux,guy=boundary.gbound_u(gux,guy,ijh,nx,ny)

    fn,gxn,gyn=mkzero.z0(fn,gxn,gyn,nx,ny)
    u_vp=stgg.u_vp_c(u_vp,u,nx,ny)
    fn,gxn,gyn=cip2d.v_cal1(vn,gvx,gvy,u_vp,v,fn,gxn,gyn,nx,ny,dx,dy,dt)
    vn,gvx,gvy=cip2d.v_cal2(fn,gxn,gyn,u_vp,v,vn,gvx,gvy,nx,ny,dx,dy,dt)
    vn=boundary.v_bound(vn,nx,ny,ijh)
    gvx,gvy=boundary.gbound_v(gvx,gvy,ijh,nx,ny)

    h=copy.copy(hn); u=copy.copy(un); v=copy.copy(vn)

    
#Time Step Update
    time=time+dt
    icount=icount+1


#1d ani = animation.ArtistAnimation(fig, ims)
#1d plt.show()
#1d ani.save('flow.gif',writer='imagemagick')
#1d ani.save('flow.mp4',writer='ffmpeg')

subprocess.call('ffmpeg -framerate 30 -i png/f%4d.png -r 60 -an -vcodec libx264 -pix_fmt yuv420p animation.mp4', shell=True)
os.system("ffmpeg -i animation.mp4 animation.gif -loop 0")