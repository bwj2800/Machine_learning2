import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

name_file = './data_lab1.txt' 

columns = ['x','y']
data_in = pd.read_csv(name_file, 
                      names=columns,
                      sep=' ')

# data_in.plot(kind='scatter',x='x',y='y',color='red')

x = np.asarray(data_in['x'])
y = np.asarray(data_in['y'])

train_x=x[:int(x.size*0.7)]
train_y=y[:int(y.size*0.7)]
test_x=x[int(x.size*0.7):]
test_y=y[int(y.size*0.7):]

train_x=np.reshape(train_x,(int(x.size*0.7),-1))
train_y=np.reshape(train_y,(int(y.size*0.7),-1))
test_x=np.reshape(test_x,(int(x.size*0.3),-1))
test_y=np.reshape(test_y,(int(y.size*0.3),-1))

# print(train_x.shape) #(70,2)
# print(train_y.shape) #(70,1)

plt.figure(5)
plt.plot(train_x,train_y,'ro')
plt.plot(test_x,test_y,'bo')
plt.xlabel('x')
plt.ylabel('y')

train_x=np.concatenate((np.ones(train_x.shape),train_x),axis=1)
test_x=np.concatenate((np.ones(test_x.shape),test_x),axis=1)

loss=1000
loss_diff=1000
epsilon=10
learning_rate=0.01
iter=0
n=1

theta=np.random.rand(n+1,1)
# print(theta.shape) #(2,1)

while(abs(loss_diff)>epsilon):
    grad=train_x.T.dot(train_x.dot(theta)-train_y)
    # print(grad.shape) #(2,1)
    theta=theta-learning_rate*grad
    
    sse=0
    for i in range(int(x.size*0.7)):
        sse+=pow((train_x[i].dot(theta)-train_y[i]),2)
    sse/=2
    loss_diff=sse-loss
    loss=sse
    
# print(theta)

sse=0
for i in range(int(x.size*0.3)):
    sse+=pow((test_x[i].dot(theta)-test_y[i]),2)
sse/=2

print(sse)
plt.plot(x,theta[0]+theta[1]*x,'g-')
plt.show()


