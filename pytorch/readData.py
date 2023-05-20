import json
import os
from typing import List

# range(25)
pickList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18]

ListLength = 3 * len(pickList)

from typing import List
import torch

markDict = {
    " ": 0,  # 无法识别的姿势
    "q": 1,  # 正确的坐姿
    "e": 2,  # 头部倾斜
    "w": 3,  # 身体前倾
    "s": 4,  # 身体后倾
    "a": 5,  # 身体右倾（对应图片中为左倾）
    "d": 6  # 身体左倾（对应图片中为右倾）
}


# len(dataList)x11x3
def sfPick(listHere):
    return [listHere[k] for k in [i * 3 + j for i in pickList for j in range(3)]]


# print(sfPick(range(75)))

# dataDir = r"../openposeOutput/json"
def readPoseData(dataDir: str) -> List[List[float]]:
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
            for i in filteredPersonIndexList:
                personX[i] = (personX[i] - pMin) / (pMax - pMin)
                personY[i] = (personY[i] - pMin) / (pMax - pMin)
                personData[i * 3] = personX[i]
                personData[i * 3 + 1] = personY[i]
            objectDataList.append(personData)
        else:
            objectDataList.append([0 for _ in range(ListLength)])
        file.close()

    print(f"总数据文件数：{len(dataNameList)}，有效非0数据文件数：{len(objectDataList)}")

    # print(objectDataList)
    return objectDataList


# "./mark.txt"
def readMark(path: str) -> List[float]:
    with open(path, "r") as file:
        markList = list(file.read())
        markList = [markDict[i] for i in markList]
        countList = [0 for i in range(7)]
        for i in markList:
            countList[i] += 1
        file.close()
    markMatrix = [[1 if j == i else 0 for j in range(7)] for i in markList]
    print(countList)
    print(torch.sum(torch.tensor(markMatrix, dtype=float)))
    return markMatrix, markList
