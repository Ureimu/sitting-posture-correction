import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import readData
import numpy

from pytorch.readMark import readMark


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(75, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 7)

    def forward(self, netX):
        netX = torch.relu(self.fc1(netX))
        netX = torch.relu(self.fc2(netX))
        netX = self.fc3(netX)
        return netX


# 定义一个空列表，用于存储每一次迭代的损失函数值
losses = []

net = Net()
markMatrix, markList = readMark("./mark.txt")
splitLimit = int(len(markMatrix)*0.7)
trainMatrix = markMatrix[0:splitLimit]
trainMarkList = markList[0:splitLimit]
testMatrix = markMatrix[splitLimit:len(markMatrix)]
testMarkList = markList[splitLimit:len(markMatrix)]

# 定义数据
x = torch.tensor(readData.readPoseData("../openposeOutput/train/json")[0:splitLimit], dtype=torch.float)
y = torch.tensor(trainMatrix, dtype=torch.float)
xTest = torch.tensor(readData.readPoseData("../openposeOutput/train/json")[splitLimit:len(markMatrix)], dtype=torch.float)
print(y)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# 训练模型
for epoch in range(1500):
    optimizer.zero_grad()
    outputs = net(x)
    loss = criterion(outputs, y.argmax(dim=1))
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

# 绘制损失函数值随着迭代次数的变化图表
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

with torch.no_grad():
    output_tensor = net(x)
    pred_probab = nn.Softmax(dim=1)(output_tensor)
    y_pred = pred_probab.argmax(1)
    print(f"trainList Predicted class: {y_pred}")
    print(f"trainList mark class: {trainMarkList}")
    print(f"accuracy: {sum([1 if y_pred[i]==trainMarkList[i] else 0 for i in range(len(y_pred))])/len(y_pred)}")

    output_tensor = net(xTest)
    pred_probab = nn.Softmax(dim=1)(output_tensor)
    y_pred = pred_probab.argmax(1)
    print(f"testList Predicted class: {y_pred}")
    print(f"testList mark class: {testMarkList}")
    print(f"accuracy: {sum([1 if y_pred[i] == testMarkList[i] else 0 for i in range(len(y_pred))]) / len(y_pred)}")

print(output_tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
