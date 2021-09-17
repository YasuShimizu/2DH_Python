import numpy as np
import copy 

nx=5
h=np.zeros([nx]);hn=np.zeros([nx])
print(id(h),id(hn))


h[:]=10.

hn=h               # hnとhのIDが共有されてしまう ---->これだとヤバい
print(id(h),id(hn))
hn=copy.copy(h)    # hnの別のIDが割り当てられる ---->OK
print(id(h),id(hn))
for i in np.arange(0,nx):  #これでも OK
    hn[i]=h[i]

hn[:]=h[:]+10     #これでもOK

print('h=',h)
print('hn=',hn)

hn[3]=100.
print('h=',h)
print('hn=',hn)


a=np.ones(5)
b=a
b[2]=100
print(a,b)
