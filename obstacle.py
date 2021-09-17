import numpy as np
import csv


def ob_ini(ijh,nx,ny):
  fopen=open('obst.dat','r')
  dataReader=csv.reader(fopen)
  d1=next(dataReader); nobst=int(d1[0])
  i1=np.zeros(nobst,dtype=int);i2=np.zeros_like(i1)
  j1=np.zeros_like(i1);j2=np.zeros_like(i1)
  for n in np.arange(0,nobst):
    lp=next(dataReader)
    i1[n]=int(lp[0]);i2[n]=int(lp[1]);j1[n]=int(lp[2]);j2[n]=int(lp[3])
#    print(i1[n],i2[n],j1[n],j2[n])
    for i in np.arange(0,nx+1):
      for j in np.arange(0,ny+1):
         if i>i1[n] and i<=i2[n] and j>j1[n] and j<=j2[n]:
          ijh[i,j]=1

  return ijh

# nobst: 障害物の個数
# i1,i2,j1,j2 : 障害物のx,y方向の範囲