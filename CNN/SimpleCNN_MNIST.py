import torch
import torch.nn as nn
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

# HyperParams
EPOCH = 10
BATCH_SIZE = 50
LR = 0.001

train_data = datasets.MNIST(
    root='../data/mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=False
)

test_data = datasets.MNIST(
    root='../data/mnist',
    train=False,
    transform=transforms.ToTensor(),
)

#  Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# two layers CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


cnn = CNN()

# loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

start_time = time.time()
# train the model
for epoch in range(EPOCH):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)

        # Forward Backward,Optimize
        # Variable.grad是累加的,因此需要置零
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()  # 更新参数
        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, EPOCH, i + 1, len(train_data) // BATCH_SIZE, loss.data[0]))

# Test the model
cnn.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * float(correct) / total))
end_time = time.time()
print("cost time:", end_time-start_time)
# Save the Trained Model
torch.save(cnn.state_dict(), '../trained_model/simple_cnn.pkl')