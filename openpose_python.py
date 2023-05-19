# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform


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
            os.environ['PATH'] = os.environ['PATH'] + ';' + rf'{openPose_path}/build/x64/Release;' + rf'{openPose_path}/build/bin;'
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
        "image_dir":    r"C:\Users\a1090\Documents\GitHub\sitting-posture-correction\pytorch\trainSet",  # 输入目录
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
