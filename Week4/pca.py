import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math

name_file = './data_pca.txt' 

columns = ['x1', 'x2']
data_in = pd.read_csv(name_file, 
                      names=columns,
                      sep=' ')
x1 = np.asarray(data_in['x1'])
x1=np.reshape(x1,(x1.size,-1))
x2 = np.asarray(data_in['x2'])
x2=np.reshape(x2,(x2.size,-1))
x=np.concatenate((x1,x2),axis=1) # x.shape=(I,N)
# print(x)

# other test data
# from sklearn import datasets
# iris=datasets.load_iris()
# iris=iris.data[:,:2]
# x1=iris[0]
# x2=iris[0]
# x=iris

plt.figure(5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(x1, x2, c="red", s=10) 

delta=1000
epsilon=0.1
I=x.shape[0] # number of training examples
N=2 # number of features
P=1 # target space dimension


#Step1: computing M
M=np.zeros((N,)) #M.shape=(N,)
for n in range(N):
    for i in range(I):
        M[n]+=x[i][n]
    M[n]/=I


#Step2: getting x_tilde
x_tilde=np.zeros(x.shape)
for i in range(I):
    for n in range(N):
        x_tilde[i][n]=x[i][n]-M[n] 


#Step3: getting covariance(=sigma)
sigma=np.zeros((N,N))
for i in range(I):
    sigma+=x_tilde[i].dot(x_tilde[i].T)
sigma/=I
# print(sigma)


#Step4: cumpute tha P-largest eigenvectors(=u) of covariance
u=np.linalg.eig(sigma)
u=u[1] #u.shape=(N,N)
u_p=np.zeros((N,P))
for p in range(P):
    u_p=u[:,p]


#Step5: project x_tilde into u_p
y_tilde_p=x_tilde.dot(u_p) #y_tilde_p.shape=(I,P)


#Step6: represent y_p in N-dimensional space
slope=u_p[1]/u_p[0]
# y=slope*(x-M[0])+M[1]
plt.plot(x, slope*(x-M[0])+M[1], 'g-')
plt.scatter(u_p[0]*y_tilde_p+M[0], u_p[1]*y_tilde_p+M[1], c="blue", s=10)

plt.show()


