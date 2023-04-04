import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

name_file = './data_kmeans.txt' 

columns = ['x','y']
data_in = pd.read_csv(name_file, 
                      names=columns,
                      sep=' ')


x = np.asarray(data_in['x'])
y = np.asarray(data_in['y'])
# print(x.shape) #(300,)

plt.figure(5)
plt.xlabel('x')
plt.ylabel('y')

delta=1000
epsilon=0
N=x.size
K=3

#초기값 설정
centroids=8*np.random.rand(K,2)
# print(centroids.shape) #(K,2)

D=np.zeros((N,K))
Y=np.zeros((N,))
while(delta>epsilon):
    for i in range(N):
        for k in range(K):
            D[i][k]=pow(x[i]-centroids[k][0],2)+pow(y[i]-centroids[k][1],2)
        Y[i]=D[i].argmin()
        
    new_centroids=np.zeros((K,2))
    for k in range(K):
        cnt=0
        for i in range(N):
            if(Y[i]==k):
                cnt+=1
                new_centroids[k][0]+=x[i]
                new_centroids[k][1]+=y[i]
        if(cnt>0):
            new_centroids[k][0]/=cnt
            new_centroids[k][1]/=cnt
      
    delta=0
    for k in range(K):
        delta+=pow(new_centroids[k][0]-centroids[k][0],2)+pow(new_centroids[k][1]-centroids[k][1],2)
        
    centroids=new_centroids
    print(delta)
        

for i in range(N):
    for k in range(K):
        D[i][k]=pow(x[i]-centroids[k][0],2)+pow(y[i]-centroids[k][1],2)
    Y[i]=D[i].argmin()
  
  
colors=['ro','bo','go','co','mo','yo','ko']
for k in range(K):
    x_temp=[]
    y_temp=[]
    for i in range(N):
        if(Y[i]==k):
            x_temp.append(x[i])
            y_temp.append(y[i])
    plt.plot(x_temp, y_temp, colors[k])
   
plt.scatter(centroids[:,0], centroids[:,1], c="black")
plt.show()

