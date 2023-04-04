import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

    
name_file = './data_ffnn_3classes.txt' 

columns = ['x1', 'x2', 'y']
data_in = pd.read_csv(name_file, 
                      names=columns,
                      sep='\t')

x1 = np.asarray(data_in['x1'])[1:].astype(np.float64)
x2 = np.asarray(data_in['x2'])[1:].astype(np.float64)
y = np.asarray(data_in['y'])[1:].astype(np.int32)

x1=np.reshape(x1,(x1.size,-1))
x2=np.reshape(x2,(x2.size,-1))
y=np.reshape(y,(y.size,-1))


loss=100000
loss_diff=100000
epsilon=10
learning_rate1=0.01
learning_rate2=0.01
I=x1.shape[0] #데이터 개수
N=2 #feature 개수
K=3 #중간 레이어 shape은 상관 X
J=3 #마지막 레이어의 shape[1]은 class개수

# weight1=v weight2=w
weight1=np.random.rand(N+1,K) #bias 더해줘서 n+1
weight2=np.random.rand(K+1, J) #bias 더해줘서 k+1

while(abs(loss_diff)>epsilon):
    #forward propagation=============================================================
    #layer1
    x_bar=np.concatenate((np.ones(x1.shape), x1, x2), axis=1) #x_bar.shape=(I,N+1)
    x_doublebar=x_bar.dot(weight1) #x_doublebar.shape=(I,K)
    f=1/(1+np.exp(-1*x_doublebar)) #f.shape=(I,K)
    #layer2
    f_bar=np.concatenate((np.ones(x1.shape), f), axis=1) #f_bar.shape=(I,K+1)
    f_doublebar=f_bar.dot(weight2) #f_doublebar.shape=(I,J)
    g=1/(1+np.exp(-1*f_doublebar)) #g.shape=(I,J)
    
    #find the labels
    # y_hat=np.ndarray((g.shape[0],1))
    # for i in range(x_bar.shape[0]):
    #     y_hat[i]=g[i].argmax()
    Y=np.zeros((I,J))
    for i in range(g.shape[0]):
        Y[i][y[i]]=1
            
    #calculate SSE(sum of squared errors)
    sse=0
    for i in range(I):
        for j in range(J):
            sse+=pow((g[i][j]-Y[i][j]),2)
    sse/=2
    loss_diff=sse-loss
    loss=sse
    # print(loss)
    
    #back propagation=============================================================
    grad=np.zeros(weight2.shape)
    for k in range(K+1):
        for j in range(J):
            for i in range(I):
                grad[k][j]+=(g[i][j]-Y[i][j])*(g[i][j])*(1-g[i][j])*(f_bar[i][k])
    weight2=weight2-learning_rate1*grad
    
    grad=np.zeros(weight1.shape)
    for n in range(N+1):
        for k in range(K):
            for i in range(I):
                for j in range(J):
                    #weight2 shape는 (K+1,J), f shape는 (I,K), k범위 1 차이
                    grad[n][k]+=(g[i][j]-Y[i][j])*(g[i][j])*(1-g[i][j])*(weight2[k+1][j])*(f[i][k])*(1-f[i][k])*(x_bar[i][n])
    weight1=weight1-learning_rate2*grad
    



