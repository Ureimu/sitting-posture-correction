import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import readData
import numpy


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(19, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, netX):
        netX = torch.relu(self.fc1(netX))
        netX = self.fc2(netX)
        return netX


# 定义一个空列表，用于存储每一次迭代的损失函数值
losses = []

net = Net()
# 定义数据
x = torch.tensor(data.readPoseData(), dtype=torch.float)
y = torch.tensor(numpy.eye(5), dtype=torch.float)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# 训练模型
for epoch in range(100):
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
    print(f"Predicted class: {y_pred}")

print(output_tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
