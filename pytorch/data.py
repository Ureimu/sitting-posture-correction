import json
import os

dataDir = r"../openposeOutput/json"
dataNameList = os.listdir(dataDir)
print(dataNameList)
objectDataList = []
for fileName in dataNameList:
    file = open(rf"{dataDir}/{fileName}", mode='r')
    data = json.load(file)
    if len(data["people"])>0:
        objectDataList.append(data["people"][0]["pose_keypoints_2d"])

print(f"总数据文件数：{len(dataNameList)}，有效数据文件数：{len(objectDataList)}")
print(objectDataList)