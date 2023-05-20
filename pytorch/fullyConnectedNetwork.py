import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import readData
import numpy

from pytorch.readData import readMark

networkDepth = 3
trainSetRate = 0.7
outputTensorLength = 7
learning_rate = 0.1
epochNum = 10000
epochReportPercentage = 0.1
MODEL_STORE_PATH = "./model"

poseData = readData.readPoseData("../openposeOutput/train/json")
print(poseData)
singlePoseDataLength = len(poseData[0])
markMatrix, markList = readMark("./mark.txt")
splitLimit = int(len(markMatrix) * trainSetRate)
trainMatrix = markMatrix[0:splitLimit]
trainMarkList = markList[0:splitLimit]
testMatrix = markMatrix[splitLimit:len(markMatrix)]
testMarkList = markList[splitLimit:len(markMatrix)]
print(f"poseLength:{singlePoseDataLength}, trainDataLength:{splitLimit}, testDataLength:{len(markMatrix) - splitLimit}")


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        intervalList = [i / networkDepth for i in range(1, networkDepth + 1)]
        intervalList.reverse()
        intervalPercentageStepIncrement = 1 / networkDepth
        startEndDelta = singlePoseDataLength - outputTensorLength
        self.fcList = [nn.Linear(int(startEndDelta * intervalPercentage + outputTensorLength),
                                 int(startEndDelta * (
                                         intervalPercentage - intervalPercentageStepIncrement) + outputTensorLength))
                       for
                       intervalPercentage in intervalList]
        self.lastFc = self.fcList.pop()

    def forward(self, netX):
        for fc in self.fcList:
            netX = torch.relu(fc(netX))
        netX = self.lastFc(netX)
        return netX


# 定义一个空列表，用于存储每一次迭代的损失函数值
losses = []
accList = []
accXList = []
accTestList = []

net = Net()

# 定义数据
x = torch.tensor(poseData[0:splitLimit], dtype=torch.float)
y = torch.tensor(trainMatrix, dtype=torch.float)
xTest = torch.tensor(poseData[splitLimit:len(markMatrix)],
                     dtype=torch.float)
yTest = torch.tensor(testMatrix, dtype=torch.float)
# print(y)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

# 训练模型

for epoch in range(1, epochNum + 1):
    optimizer.zero_grad()
    outputs = net(x)
    loss = criterion(outputs, y.argmax(dim=1))
    lossNum = loss.item()
    losses.append(lossNum)
    loss.backward()
    optimizer.step()

    if epoch % int(epochNum * epochReportPercentage) == 0 or epoch == epochNum:
        # print(outputs)
        outClass = torch.argmax(outputs, 1)
        yClass = torch.argmax(y, 1)
        # print(outClass,y)
        acc = sum([1 if yClass[i] == outClass[i] else 0 for i in range(len(y))]) / len(y)
        accList.append(acc)
        accXList.append(epoch)
        print(f"epoch:{epoch} {'%3.2f' % (epoch / epochNum * 100)}% loss:{lossNum} acc:{acc}")

        # Evaluate test set accuracy
        test_outputs = net(xTest)
        test_outClass = torch.argmax(test_outputs, 1)
        test_yClass = torch.argmax(yTest, 1)
        test_acc = sum([1 if test_yClass[i] == test_outClass[i] else 0 for i in range(len(yTest))]) / len(yTest)
        accTestList.append(test_acc)
        print(f"Test set accuracy: {test_acc}")


# 绘制损失函数值随着迭代次数的变化图表
fig = plt.figure(figsize=(7, 4))
ax1 = fig.add_subplot(111)
ax1.plot(losses, color="r", label="loss")
ax1.set_ylabel("loss")
ax1.set_xlabel("epoch")

# 绘制在训练集上的准确率随着迭代次数的变化图表
ax2 = ax1.twinx()
ax2.plot(accXList, accList, color="b", label="acc")
ax2.plot(accXList, accTestList, color="g", label="acc test")
ax2.set_xlabel("Same")
ax2.set_ylabel("accuracy on train set")


def percentageFormatter(tmp, pos):
    return '%3.1f' % (tmp * 100) + "%"


ax2.yaxis.set_major_formatter(FuncFormatter(percentageFormatter))

plt.show()

# print(output_tensor)

with torch.no_grad():
    output_tensor = net(x)
    pred_probab = nn.Softmax(dim=1)(output_tensor)
    y_pred = pred_probab.argmax(1)
    # print(f"trainList Predicted class: {y_pred}")
    # print(f"trainList mark class: {trainMarkList}")
    print(
        f"trainSet accuracy: {sum([1 if y_pred[i] == trainMarkList[i] else 0 for i in range(len(y_pred))]) / len(y_pred)}")

    output_tensor = net(xTest)
    pred_probab = nn.Softmax(dim=1)(output_tensor)
    y_pred = pred_probab.argmax(1)
    print(f"testList Predicted class: {y_pred}")
    print(f"testList mark class: {testMarkList}")
    print(
        f"testSet accuracy: {sum([1 if y_pred[i] == testMarkList[i] else 0 for i in range(len(y_pred))]) / len(y_pred)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"training finished on {device}")

torch.save(net.state_dict(), f'{MODEL_STORE_PATH}/net_model.ckpt')
