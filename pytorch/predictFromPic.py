from fullyConnectedNetwork import Net
import torch
from PIL import Image
from torchvision.transforms import transforms
from torch.autograd import Variable
import time

# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform

from pytorch.predictFromPosDataList import predictResult

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    # os.path.dirname(os.path.realpath(__file__))
    # openPose的位置
    openPose_path = r"C:/Users/a1090/Documents/GitHub/openpose"
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(rf'{openPose_path}/build/python/openpose/Release')
            os.environ['PATH'] = os.environ[
                                     'PATH'] + ';' + rf'{openPose_path}/build/x64/Release;' + rf'{openPose_path}/build/bin;'
            import pyopenpose as op
        else:
            print(f"Error: not implemented on the {platform} platform, stop importing")
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python '
            'script in the right folder?')
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = {
        "model_folder": rf"{openPose_path}\models",  # model位置
        "image_dir": r"C:\Users\a1090\Documents\GitHub\sitting-posture-correction\pytorch\trainSet",  # 输入目录
        "disable_blending": True,  # 在黑色背景上绘制骨骼坐标点。
        "write_images": r"./openposeOutput/train",  # 输出目录
        "render_threshold": 0.001,  # 控制骨骼关节点最低置信度
        "number_people_max": 1,  # 识别人数
        "write_json": r"./openposeOutput/train/json",  # JSON输出目录
        "keypoint_scale": 3  # `3` to scale it in the range [0,1], where (0,0) would be the top-left corner of the
        # image, and (1,1) the bottom-right one

    }

    poseModel = op.PoseModel.BODY_25
    print(op.getPoseBodyPartMapping(poseModel))
    print(op.getPoseNumberBodyParts(poseModel))
    print(op.getPosePartPairs(poseModel))
    print(op.getPoseMapIndex(poseModel))

    # Starting OpenPose
    opWrapper = op.WrapperPython(op.ThreadManagerMode.Synchronous)
    opWrapper.configure(params)
    opWrapper.execute()


except Exception as e:
    print(e)
    sys.exit(-1)

if __name__ == '__main__':
    start = time.process_time()
    data = [0.0, 163115.3545813507, 135018.25825455785, 12331.515849550487, 120673.87059219554, 6122.537879968295, 0.0,
            163115.3545813507, 163115.3545813507, 0.0, 0.0, 433081.6202105442, 163098.4398608161, 0.0,
            163115.3545813507, 0.4150879533697161, 0.9778420883296507, 0, 0, -0.8914324274471591, -0.2886839472216833,
            0, 0, 0, -0.6680837068163037, 0, 0, 0, 1.0, 0.5246781242226066]
    # print(predict_result(data))
    predictResult(data)

    end = time.process_time()
    print(end - start)
