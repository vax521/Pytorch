# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:21:13 2018

@author: xingxf03
"""

import torch
import torchvision
import torch.nn.functional as F
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time
import numpy as np
import matplotlib.pyplot as plt
#定义超参数
batch_size = 32
learning_rate = 1e-3
num_epochs = 100

train_data = torchvision.datasets.MNIST(
    root='../data/mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to                                               # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]                     
)
test_data = torchvision.datasets.MNIST(
          root='../data/mnist/',
          train=False,                                  
          transform=torchvision.transforms.ToTensor(), 
        )
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

# 定义 Logistic Regression模型
class Logistic_Regression(nn.Module):
    def __init__(self,in_dim,n_class):
        super(Logistic_Regression,self).__init__()
        self.logistic = nn.Linear(in_dim,n_class)
    def forward(self,x):
        out = self.logistic(x)
        return out


    
model = Logistic_Regression(28*28,10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = learning_rate)

test_acc = []
#开始训练
for epoch in range(num_epochs):
    print('*'*10)
    print('epoch:{}'.format(epoch+1))
    
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0
    for i,data in enumerate(train_loader,1):
        img,label = data
        img = img.view(img.size(0),-1) #将图片展开成28*28
        img = Variable(img)
        label = Variable(label)
        
        #前向传播
        out = model(img)
        loss = criterion(out,label)
       # print("label.size(0))
        running_loss += loss.data[0]*label.size(0)   
        _,pred = torch.max(out,1) #取出最大值的位置索引
        num_correct = (pred == label).sum()
        running_acc = num_correct.data[0]
        
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 300 ==0:
             print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epochs, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))
    print('Finish {} epoch,loss:{:.6f},Acc:{:.6f}'.format(epoch + 1, running_loss / (len(train_data)), running_acc / (len(train_data))))
        
    model.eval()
    eval_loss = 0.
    eval_acc = 0.     
    for data in test_loader:
        img,label = data
        img = img.view(img.size(0),-1)
        img = Variable(img,volatile=True)
        label = Variable(label,volatile=True)  
    
        out = model(img)
        loss = criterion(out,label)
        eval_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data[0]
    test_acc.append(eval_acc / (len(test_data)))
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / 
                 (len(test_data)), eval_acc / (len(test_data))))
        
    print('Time:{:.1f} s'.format(time.time() - since))
    print()
test_acc = np.array(test_acc)
plt.plot(test_acc)
plt.title('test_acc')
plt.show()
# 保存模型
torch.save(model.state_dict(), './logstic.pth')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

