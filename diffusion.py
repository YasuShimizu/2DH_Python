import numpy as np
from numba import jit
@jit
def diff_u(un,uvis,uvis_x,uvis_y,nx,ny,dx,dy,dt,ep,ep_x,cw):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            uvis_x[i,j]=ep[i,j]*(un[i,j]-un[i-1,j])/dx
        uvis_x[0]=uvis_x[1]; uvis_x[nx+1]=uvis_x[nx]

    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny):
            uvis_y[i,j]=ep_x[i,j]*(un[i,j+1]-un[i,j])/dy
        uvis_y[i,ny]=-cw*un[i,ny]*abs(un[i,ny])
        uvis_y[i,0]=cw*un[i,1]*abs(un[i,1])


    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            uvis[i,j]=(uvis_x[i+1,j]-uvis_x[i,j])/dx \
                +(uvis_y[i,j]-uvis_y[i,j-1])/dy
    uvis[0]=uvis[1]; uvis[nx]=uvis[nx-1]

    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny+1):
            un[i,j]=un[i,j]+uvis[i,j]*dt
    
    return un
@jit
def diff_v(vn,vvis,vvis_x,vvis_y,nx,ny,dx,dy,dt,ep,ep_x):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny):
            vvis_x[i,j]=ep_x[i,j]*(vn[i+1,j]-vn[i,j])/dx
    vvis_x[0]=vvis_x[1]; vvis_x[nx+1]=vvis_x[nx]

    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            vvis_y[i,j]=ep[i,j]*(vn[i,j]-vn[i,j-1])/dy

    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            vvis[i,j]=(vvis_x[i,j]-vvis_x[i-1,j])/dx \
                +(vvis_y[i,j+1]-vvis_y[i,j])/dy
    vvis[0]=vvis[1]; vvis[nx]=vvis[nx-1]

    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny):
            vn[i,j]=vn[i,j]+vvis[i,j]*dt
    
    return vn





