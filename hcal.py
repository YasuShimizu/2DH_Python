import numpy as np
from numba import jit
@jit
def hh(hn,h,hs,eta,qu,qv,ijh,area,alh,hmin,nx,ny,dt):
    err=0.
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            div=qu[i,j]-qu[i-1,j]+qv[i,j]-qv[i,j-1]
            hsta=h[i,j]-div/area*dt
            serr=abs(hsta-hn[i,j])
            err=err+serr
            hn[i,j]=hsta*alh+hn[i,j]*(1.-alh)
            hs[i,j]=hn[i,j]-eta[i,j]            
            if hs[i,j]<hmin:
                hs[i,j]=hmin; hn[i,j]=eta[i,j]+hmin
    return hn,hs,err                