import json
import os
from typing import List
import random
import torch

# range(25)
# [0, 1, 2, 5, 8, 9, 12, 15, 16, 17, 18]
pickList = range(25)



markDict = {
    " ": 0,  # 无法识别的姿势
    "q": 1,  # 正确的坐姿
    "e": 2,  # 头部倾斜
    "w": 3,  # 身体前倾
    "s": 4,  # 身体后倾
    "a": 5,  # 身体右倾
    "d": 6  # 身体左倾
}


# len(dataList)x11x3
def sfPick(listHere):
    return [listHere[k] for k in [i * 3 + j for i in pickList for j in range(3)]]


# print(sfPick(range(75)))
# dataDir = r"../openposeOutput/json"
def readData(markDataDir: str, dataDir: str, trainSetRate: float, useConf=True, balance=False):
    indexList = [[] for i in range(7)]
    countList = [0 for i in range(7)]
    with open(markDataDir, "r") as file:
        markList = list(file.read())
        markList = [markDict[i] for i in markList]
        index = 0
        for i in markList:
            countList[i] += 1
            indexList[i].append(index)
            index += 1
        file.close()
    markMatrix = [[1 if j == i else 0 for j in range(7)] for i in markList]

    print(torch.sum(torch.tensor(markMatrix, dtype=float)))

    # pose data
    dataNameList = os.listdir(dataDir)
    # print(dataNameList)
    objectDataList = []  # type: List[List[float]]
    for fileName in dataNameList:
        file = open(rf"{dataDir}/{fileName}", mode='r')
        data = json.load(file)
        if len(data["people"]) > 0:

            personData = sfPick(data["people"][0]["pose_keypoints_2d"])
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
            personDataWithoutConf = [0 for _ in range(2 * len(pickList))]
            for i in filteredPersonIndexList:
                personX[i] = (personX[i] - pMin) / (pMax - pMin)
                personY[i] = (personY[i] - pMin) / (pMax - pMin)
                personData[i * 3] = personX[i]
                personData[i * 3 + 1] = personY[i]
                personDataWithoutConf[i * 2] = personX[i]
                personDataWithoutConf[i * 2 + 1] = personY[i]
            if useConf:
                objectDataList.append(personData)
            else:
                objectDataList.append(personDataWithoutConf)
        else:
            if useConf:
                objectDataList.append([0 for _ in range(3 * len(pickList))])
            else:
                objectDataList.append([0 for _ in range(2 * len(pickList))])

        file.close()

    print(f"总数据文件数：{len(dataNameList)}，有效非0数据文件数：{len(objectDataList)}")

    # 平衡样本, 先保持各个类别样本数量相同（可使用balance参数指定是否执行），shuffle一次，再按照各类别分割为训练集与测试集并合并为总的训练集和测试集，再shuffle一次

    # 先保持各个类别样本数量相同，shuffle一次
    sumSize = 31
    for i in range(len(indexList)):
        if balance:
            indexList[i] = indexList[i][0:sumSize]
        random.shuffle(indexList[i])

    # 再按照各类别分割为训练集与测试集并合并为总的训练集和测试集，再shuffle一次
    print([len(i) for i in indexList])
    # 训练集
    trainIndexList = []
    for i in indexList:
        trainIndexList.extend(i[0:int(len(i)*trainSetRate)])
    random.shuffle(trainIndexList)
    shuffledTrainMarkList = []
    shuffledTrainMarkMatrix = []
    shuffledTrainObjectDataList = []
    for i in trainIndexList:
        shuffledTrainMarkList.append(markList[i])
        shuffledTrainMarkMatrix.append(markMatrix[i])
        shuffledTrainObjectDataList.append(objectDataList[i])

    # 测试集
    testIndexList = []
    for i in indexList:
        testIndexList.extend(i[int(len(i)*trainSetRate):len(i)])
    random.shuffle(testIndexList)
    shuffledTestMarkList = []
    shuffledTestMarkMatrix = []
    shuffledTestObjectDataList = []
    for i in testIndexList:
        shuffledTestMarkList.append(markList[i])
        shuffledTestMarkMatrix.append(markMatrix[i])
        shuffledTestObjectDataList.append(objectDataList[i])

    return shuffledTrainMarkList, shuffledTrainMarkMatrix, shuffledTrainObjectDataList, shuffledTestMarkList, shuffledTestMarkMatrix, shuffledTestObjectDataList, indexList
