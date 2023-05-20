from typing import List

markDict = {
    " ": 0,  # 无法识别的姿势
    "q": 1,  # 正确的坐姿
    "e": 2,  # 头部倾斜
    "w": 3,  # 身体前倾
    "s": 4,  # 身体后倾
    "a": 5,  # 身体右倾（对应图片中为左倾）
    "d": 6  # 身体左倾（对应图片中为右倾）
}


# "./mark.txt"
def readMark(path: str) -> List[float]:
    with open(path, "r") as file:
        markList = list(file.read())
        markList = [markDict[i] for i in markList]
    markMatrix = [[1 if j == i else 0 for j in range(7)] for i in markList]
    return markMatrix,markList
