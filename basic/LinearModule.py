# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:34:20 2018

@author: xingxf03
"""

import torch
from torch import nn,optim 
from  torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

#LinearModule
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(1,1,bias=True)
    def forward(self,x):
        out = self.linear(x)
        return out
model = LinearRegression()


criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.0001)
num_epochs = 1000
loss_num=[]

for epoch in range(num_epochs):
     inputs = Variable(x_train)
     target = Variable(y_train)
     
     #forward
     out = model(inputs)
     loss = criterion(out,target)
     
     #backward
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()
     
     if(epoch+1)%20 ==0:
         loss_num.append(float(loss.data[0]))
         print('Epoch:[{}/{}], loss: {}'.format(epoch+1,num_epochs,loss.data[0]))
model.eval()
predict = model(Variable(x_train))
predict = predict.data.numpy()

loss_num=np.array(loss_num)

plt.subplot(121)
plt.plot(loss_num)
plt.ylabel('loss')

plt.subplot(122)
plt.plot(x_train.numpy(),y_train.numpy(),'ro',label='Original data')
plt.plot(x_train.numpy(), predict, label='Fitting Line')  
  
plt.legend(loc='best')
plt.show()

torch.save(model.state_dict(),'./linear.pth')



















     
     