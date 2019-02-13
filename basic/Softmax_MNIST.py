
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time
import numpy as np
import matplotlib.pyplot as plt
# 定义超参数
batch_size = 32
learning_rate = 1e-3
num_epoches = 100

# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='../data/mnist', train=True, transform=transforms.ToTensor())

test_dataset = datasets.MNIST(
    root='../data/mnist', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 定义 Logistic Regression 模型
class Softmax_NN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Softmax_NN, self).__init__()
        self.layer1 = nn.Linear(in_dim, 520)
        self.layer2 = nn.Linear(520,320)
        self.layer3 = nn.Linear(320,240)
        self.layer4 = nn.Linear(240,120)
        self.layer5 = nn.Linear(120,n_class)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return F.log_softmax(self.layer5(x), dim=1)


model = Softmax_NN(28 * 28, 10)  # 图片大小是28x28
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.5)


test_acc = []
# 开始训练
for epoch in range(num_epoches):
    print('*' * 10)
    print('epoch {}'.format(epoch + 1))
    
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.view(img.size(0), -1)  # 将图片展开成 28x28
        if use_gpu:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data[0]
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))
    
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if use_gpu:
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data[0]
    test_acc.append(eval_acc / (len(test_dataset)))
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print('Time:{:.1f} s'.format(time.time() - since))
    print()
test_acc = np.array(test_acc)
plt.plot(test_acc)
plt.title('test_acc')
plt.show()
# 保存模型
# 保存模型
# #准确率 97.55%
torch.save(model.state_dict(), './logstic.pth')
