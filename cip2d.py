import numpy as np
import copy
from numba import jit
@jit
def u_cal1(un,gux,guy,u,v_up,fn,gxn,gyn,nx,ny,dx,dy,dt):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            xx=-u[i,j]*dt;  yy=-v_up[i,j]*dt
            isn=int(np.sign(u[i,j])); jsn=int(np.sign(v_up[i,j]))
            if isn==0:
                isn=1
            if jsn==0:
                jsn=1
            im=i-isn; jm=j-jsn
            a1=((gux[im,j]+gux[i,j])*dx*isn-2.*(un[i,j]-un[im,j]))/(dx**3*isn)
            b1=((guy[i,jm]+guy[i,j])*dy*jsn-2.*(un[i,j]-un[i,jm]))/(dy**3*jsn)
            e1=(3.*(un[im,j]-un[i,j])+(gux[im,j]+2.*gux[i,j])*dx*isn)/dx**2
            f1=(3.*(un[i,jm]-un[i,j])+(guy[i,jm]+2.*guy[i,j])*dy*jsn)/dy**2
            tmp=un[i,j]-un[i,jm]-un[im,j]+un[im,jm]
            tmq1=guy[im,j]-guy[i,j]
            tmp2=gux[i,jm]-gux[i,j]
            c1=(-tmp-tmp2*dx*isn)/(dx**2*dy*jsn)
            d1=(-tmp-tmq1*dy*jsn)/(dx*dy**2*isn)
            g1=(-tmq1+c1*dx**2)/(dx*isn)
            fn[i,j]=((a1*xx+c1*yy+e1)*xx+g1*yy+gux[i,j])*xx \
              +((b1*yy+d1*xx+f1)*yy+guy[i,j])*yy+un[i,j]            
            gxn[i,j]=(3.*a1*xx+2.*(c1*yy+e1))*xx+(d1*yy+g1)*yy+gux[i,j]
            gyn[i,j]=(3.*b1*yy+2.*(d1*xx+f1))*yy+(c1*xx+g1)*xx+guy[i,j]
    return fn,gxn,gyn
@jit
def u_cal2(fn,gxn,gyn,u,v_up,un,gux,guy,nx,ny,dx,dy,dt):
    un=fn.copy()   #これは上手く行く
    gux=gxn.copy()
    guy=gyn.copy()
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            gxo=(u[i+1,j]-u[i-1,j])*.5/dx
            gyo=(u[i,j+1]-u[i,j-1])*.5/dy
            gux[i,j]=gux[i,j]-(gxo*(u[i+1,j]-u[i-1,j])+gyo*(v_up[i+1,j]-v_up[i-1,j]))*.5*dt/dx
            guy[i,j]=guy[i,j]-(gxo*(u[i,j+1]-u[i,j-1])+gyo*(v_up[i,j+1]-v_up[i,j-1]))*.5*dt/dy
    return un,gux,guy
@jit
def v_cal1(vn,gvx,gvy,u_vp,v,fn,gxn,gyn,nx,ny,dx,dy,dt):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny):
            xx= -u_vp[i,j]*dt; yy= -v[i,j]*dt
            isn=int(np.sign(u_vp[i,j]));  jsn=int(np.sign(v[i,j]))
            if isn==0:
                isn=1
            if jsn==0:
                jsn=1
            im=i-isn; jm=j-jsn
            a1=((gvx[im,j]+gvx[i,j])*dx*isn-2.*(vn[i,j]-vn[im,j]))/(dx**3*isn)
            b1=((gvy[i,jm]+gvy[i,j])*dy*jsn-2.*(vn[i,j]-vn[i,jm]))/(dy**3*jsn)
            e1=(3.*(vn[im,j]-vn[i,j])+(gvx[im,j]+2.*gvx[i,j])*dx*isn)/dx**2
            f1=(3.*(vn[i,jm]-vn[i,j])+(gvx[i,jm]+2.*gvy[i,j])*dy*jsn)/dy**2
            tmp=vn[i,j]-vn[i,jm]-vn[im,j]+vn[im,jm]
            tmq1=gvy[im,j]-gvy[i,j]
            tmq2=gvx[i,jm]-gvx[i,j]
            c1=(-tmp-tmq2*dx*isn)/(dx**2*dy*jsn)
            d1=(-tmp-tmq1*dy*jsn)/(dx*dy**2*isn)
            g1=(-tmq1+c1*dx**2)/(dx*isn)
            fn[i,j]=((a1*xx+c1*yy+e1)*xx+g1*yy+gvx[i,j])*xx  \
             +((b1*yy+d1*xx+f1)*yy+gvy[i,j])*yy+vn[i,j]
            gxn[i,j]=(3.*a1*xx+2.*(c1*yy+e1))*xx+(d1*yy+g1)*yy+gvx[i,j]
            gyn[i,j]=(3.*b1*yy+2.*(d1*xx+f1))*yy+(c1*xx+g1)*xx+gvy[i,j]
    return fn,gxn,gyn
@jit
def v_cal2(fn,gxn,gyn,u_vp,v,vn,gvx,gvy,nx,ny,dx,dy,dt): 
    vn=fn.copy()
    gvx=gxn.copy()
    gvy=gyn.copy()
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny):
            gxo=(v[i+1,j]-v[i-1,j])*.5/dx
            gyo=(v[i,j+1]-v[i,j-1])*.5/dy
            gvx[i,j]=gvx[i,j]-(gxo*(u_vp[i+1,j]-u_vp[i-1,j])+gyo*(v[i+1,j]-v[i-1,j]))*0.5*dt/dx
            gvy[i,j]=gvy[i,j]-(gxo*(u_vp[i,j+1]-u_vp[i,j-1])+gyo*(v[i,j+1]-v[i,j-1]))*0.5*dt/dy
    return vn,gvx,gvy