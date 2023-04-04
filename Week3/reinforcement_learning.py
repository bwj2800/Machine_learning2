import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

V=np.full((4,3),0.0)
#V 초기화 1, -1??
V[3][2]=1
V[3][1]=-1
PI=np.full((4,3),'')

def calculatePV(x,y,dir):
    n=0
    e=0
    w=0
    if dir==0: #북
        if y<2 and not(x==1 and (y+1)==1): #직진
            n=0.8*V[x][y+1]
        else: 
            n=0.8*V[x][y]
        
        if x<3 and not((x+1)==1 and y==1): #우회전
            e=0.1*V[x+1][y]
        else:
            e=0.1*V[x][y]
        
        if x>0 and not((x-1)==1 and y==1): #좌회전
            w=0.1*V[x-1][y]
        else:
            w=0.1*V[x][y]
    elif dir==1: #남
        if y>1 and not(x==1 and (y-1)==1): #직진
            n=0.8*V[x][y-1]
        else:
            n=0.8*V[x][y]
            
        if x>0 and not((x-1)==1 and y==1): #우회전
            e=0.1*V[x-1][y]
        else:
            e=0.1*V[x][y]
        
        if x<3 and not((x+1)==1 and y==1): #좌회전
            w=0.1*V[x+1][y]
        else: 
            w=0.1*V[x][y]
    elif dir==2: #동
        if x<3 and not((x+1)==1 and y==1): #직진
            n=0.8*V[x+1][y]
        else:
            n=0.8*V[x][y]
            
        if y>0 and not(x==1 and (y-1)==1): #우회전
            e=0.1*V[x][y-1]
        else:
            e=0.1*V[x][y]
        
        if y<2 and not(x==1 and (y+1)==1): #좌회전
            w=0.1*V[x][y+1]
        else: 
            w=0.1*V[x][y]
    else: #서
        if x>0 and not((x-1)==1 and y==1): #직진
            n=0.8*V[x-1][y]
        else:
            n=0.8*V[x][y]
        
        if y<2 and not(x==1 and (y+1)==1): #우회진
            e=0.1*V[x][y+1]
        else: 
            e=0.1*V[x][y]
        
        if y>0 and not(x==1 and (y-1)==1): #좌회전
            w=0.1*V[x][y-1]
        else:
            w=0.1*V[x][y]
    
    return n+w+e

reward=np.full((4,3),-0.02)
reward[3][2]=1
reward[3][1]=-1
discount_factor=0.5
A=['N', 'S', 'E', 'W']
#P=[0.8, 0, 0.1, 0.1]

#V 차이, epsilon 비교로 수렴 여부 판단 가능
for iter in range(30):
    for x in range(4):
        for y in range(3):
            PV=np.zeros((4))
            for i in range(4):
                PV[i]=calculatePV(x,y,i)
            
            if not((x==3 and y==2) or (x==3 and y==1) or (x==1 and y==1)):
                V[x][y]=reward[x][y]+discount_factor*max(PV)
            PI[x][y]=A[PV.argmax()]
        

print("Value=====")
for i in range(3):
    for j in range(4):
        print('{0:<10} '.format(round(V[j][2-i],3)),end=' ')
        # print((V[j][2-i]),end=' ')
    print()
    
print("Policy=====")
for i in range(3):
    for j in range(4):
        print((PI[j][2-i]),end=' ')
    print()
    
    