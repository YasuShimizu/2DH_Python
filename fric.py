import numpy as np

def us_cal(usta,ep,ep_x,u,v,hs,nx,ny,snm,g_sqrt,hmin,ep_alpha):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            ux=(u[i,j]+u[i-1,j])*.5
            vx=(v[i,j]+v[i,j-1])*.5
            vv=np.sqrt(ux**2+vx**2)
            if hs[i,j]>hmin:
                usta[i,j]=snm*g_sqrt*vv/hs[i,j]**(1./6.)
                ep[i,j]=.4/6.*usta[i,j]*hs[i,j]*ep_alpha
            else:
                usta[i,j]=0.
                ep[i,j]=0.
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            ep_x[i,j]=(ep[i,j]+ep[i+1,j]+ep[i,j+1]+ep[i+1,j+1])*.25
    return usta,ep,ep_x


