import numpy as np

def uv(ux,vx,uv2,hx,hsx,u,v,h,hs,nx,ny):
    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny):
            ux[i,j]=(u[i,j]+u[i,j+1])*.5
        ux[i,0]=u[i,1]; ux[i,ny]=u[i,ny]
    
    
    for j in np.arange(0,ny+1):
        for i in np.arange(1,nx):    
            vx[i,j]=(v[i,j]+v[i+1,j])*.5
        vx[0,j]=v[1,j]; vx[nx,j]=v[nx,j]

    uv2[:,:]=np.sqrt(ux[:,:]**2+vx[:,:]**2)

    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny):
            hx[i,j]=(h[i,j]+h[i+1,j]+h[i,j+1]+h[i+1,j+1])*.25
            hsx[i,j]=(hs[i,j]+hs[i+1,j]+hs[i,j+1]+hs[i+1,j+1])*.25
        hx[i,0]=(h[i,j]+h[i+1,j])*.5
        hx[i,ny]=(h[i,ny]+h[i+1,ny])*.5
        hsx[i,0]=(hs[i,j]+hs[i+1,j])*.5
        hsx[i,ny]=(hs[i,ny]+hs[i+1,ny])*.5
    
    return ux,vx,uv2,hx,hsx

def vortex(vor,ux,vx,nx,ny,dx,dy):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            vor[i,j]=(ux[i,j+1]-ux[i,j-1])/(2.*dy)- \
                     (vx[i+1,j]-vx[i-1,j])/(2.*dx)

    return vor

