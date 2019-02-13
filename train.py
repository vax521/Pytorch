# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 08:59:32 2018

@author: xingxf03
"""

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets,models,transforms
import time 
import os

def train_model(model,certerion,optimizer,scheduler,num_epochs=25):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch,num_epochs-1))
        print('-'*10)
        
        #Each epoch has a traning and validation phase
        for phase in['train','val']:
            if(phase == 'train'):
                scheduler.step()
                model.train(True) #set model to training mode 
            else:
                model.train(False)
                
            running_loss = 0.0
            running_corrects = 0
                
                #Iterate over data
            for data in dataloaders[phase]:
                    inputs,labels = data
                    #wrap them in Variable
                    if(user_gpu):
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs,labels = Variable(inputs),Variable(labels)
                      
                    #权值初始化为0
                    optimizer.zero_grad()
                    
                    #forward
                    outputs = model(inputs)
                    _,preds = torch.max(outputs.data,1)
                    loss =  certerion(outputs,labels)
                   #backword+optimizer only if in training phase
                    if(phase == 'train'):
                        loss.backward()
                        optimizer.step()
                    #statistics
                    running_loss +=loss.data[0]
                    running_corrects += torch.sum(preds = labels.data)
                     
            epoch_loss = running_loss/dataset_size[phase]
            epoch_acc = running_corrects/dataset_size[phase]
             
            print('{} loss:{:.4f} Acc:{:.4f}'.format(phase,epoch_loss,epoch_acc))
                        
            if phase == 'val' and epoch_acc >best_acc:
                  best_acc = epoch_acc
                  best_model_wts = model.state_dict()
    
    time_elapsed = time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60,time_elapsed%60))
    print('Best val Acc:{:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model
                  
if __name__ == '__main__' :
     data_transforms = {
             'train':transforms.Compose([
                     transforms.RandomSizedCrop(224),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ]),
             'val':transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                     ])
             }    
     data_dir = '/data'
     image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),
                                              data_transform[x] for x in ['train','val']
                                             )}
     dataloaders = { x:torch.utils.data.DataLoader(
                                          image_datasets[x],
                                          batch_size = 4,
                                          num_workers= 4) for x in ['train','val']
                                               }
     dataset_sizes = {x:len(image_dataset[x])for x in ['train','val']}
    
    # use gpu or not
     use_gpu = torch.cuda.is_available()
    
    #get the model and replace the original fc layer with your fc layer
     model_ft = models.resnet18(pretrained=True)
     num_ftrs = model_ft.fc.in_features
     model_ft.fc = nn.Linear(num_ftrs,2)
    
     if use_gpu:
        model_ft = model_ft.cuda()
    
     criterion = nn.CrossEntropyLoss()
    
     optimizer_ft = optim.SGD(model_fit.parameters(),lr=0.001,momentum=0.9)
    
     exp_lr_schedular = lr_scheduler.StepLR(optimizer_ft,step_size=7,gamma=0.1)
    
     model_ft = train_model(
             model=model_ft,
             criterion=criterion,
             optimizer = optimizer_ft,
             schedular = exp_lr_schedular,
             num_epochs = 25
            )
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        