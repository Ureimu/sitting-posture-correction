import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from readData import readData
import numpy

networkDepth = 4  # 网络深度
# [75, 1000, 1800, 1200, 7]
networkSizeList = [75, 1000, 1800, 1200, 7]
useExponent = False  # 是否使用指数参数控制
exponent = 0.7  # 指数参数，控制网络每层的值是偏大还是偏小。
outputTensorLength = 7  # 7分类问题。
learning_rate = 5  # 学习率
epochNum = 100000  # 训练轮数, 40000
# 每隔step_size个epoch，学习率 x gamma
step_size = 50
gamma = 1

printAcc = True  # 是否打印acc参数
epochReportPercentage = 0.01  # acc参数在每完成百分之多少的时候输出

model_store_path = "./model"  # 模型存储路径

trainMarkList, trainMatrix, trainPoseData, testMarkList, testMatrix, testPoseData, indexList = readData("./mark.txt",
                                                                                                        "../openposeOutput/train/json",
                                                                                                        1,
                                                                                                        True)
trainClassNumList = []
testClassNumList = []
for i in range(outputTensorLength):
    trainClassNumList.append(sum([1 if i == j else 0 for j in trainMarkList]))
    testClassNumList.append(sum([1 if i == j else 0 for j in testMarkList]))
print(f"train data class num {trainClassNumList}")
# print(poseData)
singlePoseDataLength = len(trainPoseData[0])

print(f"poseLength:{singlePoseDataLength}, trainDataLength:{len(trainPoseData)}")


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        startEndDelta = singlePoseDataLength - outputTensorLength
        intervalList = []
        if useExponent:
            intervalList = [(i ** exponent) / (networkDepth ** exponent) for i in range(0, networkDepth + 1)]
            intervalList.reverse()
            print(
                f"network size List: {[int(startEndDelta * intervalPercentage + outputTensorLength) for intervalPercentage in intervalList]}")

        else:
            intervalList = networkSizeList
            print(
                f"network size List: {intervalList}")

        self.fcList = []
        for i in range(len(intervalList) - 1):
            if useExponent:
                intervalPercentage = intervalList[i]
                nextPercentage = intervalList[i + 1]
                self.fcList.append(nn.Linear(int(startEndDelta * intervalPercentage + outputTensorLength),
                                             int(startEndDelta * nextPercentage + outputTensorLength)))
            else:
                intervalPercentage = intervalList[i]
                nextPercentage = intervalList[i + 1]
                self.fcList.append(nn.Linear(intervalPercentage,
                                             nextPercentage))

        self.lastFc = self.fcList.pop()

    def forward(self, netX):
        for fc in self.fcList:
            netX = torch.relu(fc(netX))
        netX = self.lastFc(netX)
        return netX


if __name__ == '__main__':

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
    # criterion = nn.CrossEntropyLoss()
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


            if printAcc:
                print(f"epoch:{epoch} {'%3.2f' % (epoch / epochNum * 100)}% loss:{lossNum} acc:{acc}")

    # 绘制损失函数值随着迭代次数的变化图表
    fig = plt.figure(figsize=(6, 6))

    ax1 = fig.add_subplot(111)
    ax1.plot(losses, color="r", label="loss")
    ax1.set_ylabel("loss")
    ax1.set_xlabel("epoch")

    # 绘制在训练集上的准确率随着迭代次数的变化图表
    ax2 = ax1.twinx()
    ax2.plot(accXList, accList, color="b", label="acc")
    ax2.set_xlabel("Same")
    ax2.set_ylabel("accuracy")

    fig.legend(labels=('loss', 'acc'), loc="upper right")


    def percentageFormatter(tmp, pos):
        return '%3.1f' % (tmp * 100) + "%"


    ax2.yaxis.set_major_formatter(FuncFormatter(percentageFormatter))

    plt.show()

    fig.savefig('./chart/trainLossChart.png')

    # print(output_tensor)

    with torch.no_grad():
        output_tensor = net(x)
        pred_probab = nn.Softmax(dim=1)(output_tensor)
        y_pred = pred_probab.argmax(1)
        # print(f"trainList Predicted class: {y_pred}")
        # print(f"trainList mark class: {trainMarkList}")
        print(
            f"trainSet accuracy: {sum([1 if y_pred[i] == trainMarkList[i] else 0 for i in range(len(y_pred))]) / len(y_pred)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training finished on {device}")

    torch.save(net.state_dict(), f'{model_store_path}/net_model.ckpt')
