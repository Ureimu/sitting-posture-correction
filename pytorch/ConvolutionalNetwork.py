import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from readData import readData
import numpy

networkDepth = 3  # 网络深度
trainSetRate = 0.6  # 测试集占总数据集的比例
outputTensorLength = 7  # 7分类问题。
learning_rate = 5e-3  # 学习率
epochNum = 10000  # 训练轮数
# 每隔step_size个epoch，学习率 x gamma
step_size = 50
gamma = 1

printAcc = False  # 是否打印acc参数
epochReportPercentage = 0.01  # acc参数在每完成百分之多少的时候输出
exponent = 1  # 指数参数，控制网络每层的值是偏大还是偏小。
model_store_path = "./model"  # 模型存储路径

trainMarkList, trainMatrix, trainPoseData, testMarkList, testMatrix, testPoseData, indexList = readData("./mark.txt",
                                                                                                        "../openposeOutput/train/json",
                                                                                                        trainSetRate,
                                                                                                        False)
trainClassNumList = []
testClassNumList = []
for i in range(outputTensorLength):
    trainClassNumList.append(sum([1 if i == j else 0 for j in trainMarkList]))
    testClassNumList.append(sum([1 if i == j else 0 for j in testMarkList]))
print(f"train data class num {trainClassNumList}, test data class num {testClassNumList}")
# print(poseData)
singlePoseDataLength = len(trainPoseData[0])

print(f"poseLength:{singlePoseDataLength}, trainDataLength:{len(trainPoseData)}, testDataLength:{len(testPoseData)}")


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        intervalList = [(i ** exponent) / (networkDepth ** exponent) for i in range(0, networkDepth + 1)]
        intervalList.reverse()

        startEndDelta = singlePoseDataLength - outputTensorLength
        print(
            f"network size List: {[int(startEndDelta * intervalPercentage + outputTensorLength) for intervalPercentage in intervalList]}")

        self.fcList = []
        for i in range(len(intervalList) - 1):
            intervalPercentage = intervalList[i]
            nextPercentage = intervalList[i + 1]
            self.fcList.append(nn.Linear(int(startEndDelta * intervalPercentage + outputTensorLength),
                                         int(startEndDelta * nextPercentage + outputTensorLength)))
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
x = torch.tensor(trainPoseData, dtype=torch.float)
y = torch.tensor(trainMatrix, dtype=torch.float)
# print(testMatrix)
xTest = torch.tensor(testPoseData,
                     dtype=torch.float)
yTest = torch.tensor(testMatrix, dtype=torch.float)
# print(y)

maxClassNum = max(trainClassNumList)
weights = torch.FloatTensor([maxClassNum / i for i in trainClassNumList])  # 类别权重，与样本数成反比
# 定义损失函数和优化器

criterion = nn.CrossEntropyLoss(weights)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

# 定义学习率更新。该方法是：每隔step_size个epoch，学习率 x gamma
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
# 训练模型

for epoch in range(1, epochNum + 1):
    optimizer.zero_grad()
    outputs = net(x)
    loss = criterion(outputs, y.argmax(dim=1))
    lossNum = loss.item()
    losses.append(lossNum)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % int(epochNum * epochReportPercentage) == 0 or epoch == epochNum:
        # print(outputs)
        outClass = torch.argmax(outputs, 1)
        yClass = torch.argmax(y, 1)
        # print(outClass,y)
        acc = sum([1 if yClass[i] == outClass[i] else 0 for i in range(len(y))]) / len(y)
        accList.append(acc)
        accXList.append(epoch)

        # Evaluate test set accuracy
        test_outputs = net(xTest)
        test_outClass = torch.argmax(test_outputs, 1)
        test_yClass = torch.argmax(yTest, 1)
        test_acc = sum([1 if test_yClass[i] == test_outClass[i] else 0 for i in range(len(yTest))]) / len(yTest)
        accTestList.append(test_acc)

        if printAcc:
            print(f"epoch:{epoch} {'%3.2f' % (epoch / epochNum * 100)}% loss:{lossNum} acc:{acc}")
            print(f"Test set accuracy: {test_acc}")

# 绘制损失函数值随着迭代次数的变化图表
fig = plt.figure(figsize=(6, 6))

ax1 = fig.add_subplot(111)
ax1.plot(losses, color="r", label="loss")
ax1.set_ylabel("loss")
ax1.set_xlabel("epoch")

# 绘制在训练集上的准确率随着迭代次数的变化图表
ax2 = ax1.twinx()
ax2.plot(accXList, accList, color="b", label="train acc")
ax2.plot(accXList, accTestList, color="g", label="test acc")
ax2.set_xlabel("Same")
ax2.set_ylabel("accuracy on train set")

fig.legend(labels=('loss', 'train acc', "test acc"), loc="upper right")


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

torch.save(net.state_dict(), f'{model_store_path}/net_model.ckpt')
