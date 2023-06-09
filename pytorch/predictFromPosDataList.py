import torch.nn as nn
from torch.autograd import Variable
import torch


singlePoseDataLength = 75
outputTensorLength = 7
networkSizeList = [75, 1000, 1800, 1200, 7]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        startEndDelta = singlePoseDataLength - outputTensorLength
        intervalList = []

        intervalList = networkSizeList
        print(
            f"network size List: {intervalList}")

        self.fcList = []
        for i in range(len(intervalList) - 1):
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


model = None
if not model:
    model = Net()
    model.load_state_dict(torch.load(
        fr"C:\Users\a1090\Documents\GitHub\sitting-posture-correction\pytorch\model\net_model4.ckpt", map_location='cpu'))


def predictResult(datas=None) -> int:
    """
    :param datas: list
    :return: int
    """
    if datas is None:
        datas = []

    personData = torch.Tensor(datas).float().numpy().tolist()
    # print(personData)
    personX = personData[0:len(personData):3]
    personY = personData[1:len(personData):3]
    personConf = personData[2:len(personData):3]
    filteredPersonCoordList = []
    filteredPersonIndexList = []
    for i in range(len(personConf)):
        if personConf[i] != 0:
            filteredPersonCoordList.append(personX[i])
            filteredPersonCoordList.append(personY[i])
            filteredPersonIndexList.append(i)
    pMax = max(filteredPersonCoordList)
    pMin = min(filteredPersonCoordList)
    for i in filteredPersonIndexList:
        personX[i] = (personX[i] - pMin) / (pMax - pMin)
        personY[i] = (personY[i] - pMin) / (pMax - pMin)
        personData[i * 3] = personX[i]
        personData[i * 3 + 1] = personY[i]
    predict = model(Variable(torch.Tensor([personData]).float())).detach().cpu().numpy().tolist()[0]
    predict = predict.index(max(predict))

    return predict
