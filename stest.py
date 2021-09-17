import numpy as np

v=np.zeros([6,6]);v_up=np.zeros([6,6])
print(len(v))
v[3,3]=5

print('v=',v)

for i in np.arange(0,5):
    print('i=',i)
    for j in np.arange(1,6):
            v_up[i,j]=(v[i,j]+v[i,j-1]+v[i+1,j]+v[i+1,j-1])*.25

print('v_up=',v_up)

v_up[:-1,1:]=(v[:-1,1:]+v[:-1,:-1]+v[1:,1:]+v[1:,:-1])*.25

print('v_up=',v_up)


