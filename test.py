import numpy as np
from numba import njit, jit, f8

u=np.zeros([5,4])
f=np.zeros([5,4])

def calc(u):
    for i in np.arange(0,5):
        for j in np.arange(0,4):
            u[i,j]=2.
            f[i,j]=u[i,j]**2
            u[i,j]=f[i,j]*2+u[i,j]
    return u

u=calc(u)
print('u=',u,'f=',f)

print("f = {}", id(f))
print("u = {}", id(u))

print(id(u),id(f))



