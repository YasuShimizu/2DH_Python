import numpy as np
import csv
nx=40;ny=40
ijh=np.zeros([nx+2,ny+2],dtype=int)
f=open('obst.dat','r')
dataReader=csv.reader(f)
a1=next(dataReader)
nobst=int(a1[0])
i1=np.zeros(nobst,dtype=int);i2=np.zeros_like(i1)
j1=np.zeros_like(i1);j2=np.zeros_like(i1)
for n in np.arange(0,nobst):
    l=next(dataReader)
    i1[n]=int(l[0]);i2[n]=int(l[1]);j1[n]=int(l[2]);j2[n]=int(l[3])
    print(i1[n],i2[n],j1[n],j2[n])
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            if i>i1[n] and i<=i2[n] and j>j1[n] and j<=j2[n]:
                ijh[i,j]=1
print(ijh[1:-1,1:-1])







