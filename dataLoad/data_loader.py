# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 16:15:37 2018

@author: xingxf03
"""
#import torch
#import torch.utils.data as Data
#
#torch.manual_seed(1)
#BATCH_SIZE = 5
#
#x = torch.linspace(1,10,10)
#y = torch.linspace(10,1,10)
#
#torch_dataset = Data.TensorDataset(data_tensor=x,target_tensor=y)
#
#data_loader = Data.DataLoader(
#        dataset = torch_dataset,
#        batch_size=BATCH_SIZE,
#        shuffle=True,  #是否打乱顺序
#        num_workers=2 #多线程读取数据
#        )
#for epoch in range(3):
#    for step,(batch_x,batch_y) in enumerate(data_loader):
#        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
#              batch_x.numpy(), '| batch y: ', batch_y.numpy())

import torch
import torch.utils.data as Data

torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5
# BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)

torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

for epoch in range(3):   # train entire dataset 3 times
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
        # train your data...
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())
