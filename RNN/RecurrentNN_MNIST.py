# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 09:11:56 2018

@author: xingxf03
"""
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time
import numpy as np 
import matplotlib.pyplot as plt

# 超参数
batch_size = 100
learning_rate = 1e-3
num_epoches = 40

# 加载数据集
train_data = datasets.MNIST(root='../data/mnist',
                            train=True,
                            transform=transforms.ToTensor())
test_data = datasets.MNIST(root='../data/mnist',
                           train=False,
                           transform=transforms.ToTensor()
                           )

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# 将图片看成长度为28的序列，序列中的每个数据的维度是28
class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(RNN, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        # LSTM输入维度是28，输出维度是128
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        out, _ = self.lstm(x)
        # out中的三个维度分别表示batch_size,序列长度，数据维度
        # 中间的序列长度取-1，表示取序列中的最后一个数据，
        # 这个数据维度为128，再通过分类器，输出10个结果表示每种结果的概率。
        out = out[:, -1, :]
        out = self.classifier(out)
        return out


model = RNN(28, 128, 1, 10)

# 定义损失函数和梯度下降算法
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

test_loss = []
test_acc = []
# 训练过程
for epoch in range(num_epoches):
    print('*'*10)
    print('epoch:{}'.format(epoch+1))
    
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        b, c, h, w = img.size()
        assert c == 1, 'channel must be 1'
        img = img.squeeze(1)
        
        img = Variable(img)
        label = Variable(label)
        
        # 前向传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data[0]*label.size(0)
        # 维数参数的时候，它的返回值为两个，一个为最大值，另一个为最大值的索引
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data[0]
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print('[{}/{}] Loss:{:.6f},Acc:{:.6f}'.format(
                    epoch+1, num_epoches,
                    running_loss/(batch_size*i),
                    running_acc/(batch_size*i)
                    ))
    print('Finish {} epoch,loss:{:.6f},Acc:{:.6f}'.format(
            epoch+1, running_loss/(len(train_data)),
            running_acc/(len(train_data))
            ))  
    
    # 验证模型
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for data in test_loader:
        img, label = data
        
        b, c, h, w = img.size()  # 100,1,28,28
        
        assert c == 1, 'channel must be 1'
        img = img.squeeze(1)
        
        img = Variable(img, volatile=True)
        # 拼成了lable，导致与前面的
        label = Variable(label, volatile=True)
        
        out = model(img)
        loss = criterion(out,label)
        eval_loss += loss.data[0]*label.size(0)
        _,pred = torch.max(out,1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data[0]
    
    test_loss.append(eval_loss/(len(test_data)))
    test_acc.append(eval_acc/(len(test_data)))
    print('Test Loss:{:.6f},Test Acc:{:.6f}'.format(
            eval_loss/(len(test_data)),
            eval_acc/(len(test_data))
            ))
    print('Time:{:.1f} s'.format(time.time() - since))
    print()
test_loss = np.array(test_loss)
test_acc = np.array(test_acc)
plt.subplot(211)
plt.plot(test_loss, label="test_loss")
plt.subplot(212)
plt.plot(test_acc, label='test_acc')
torch.save(model.state_dict, '../trained_model/rnn_mnisst.pth')

# 20个epoch n_layer=2 acc:9836
# 40个epoch n_layer=2 acc:9896
# 40个epoch n_layer=1 acc:0.9900
