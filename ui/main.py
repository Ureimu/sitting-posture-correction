import os
import sys
import cv2
from PyQt5.QtCore import QTimer, QCoreApplication
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame

from pytorch.predictFromPosDataList import predictResult

# 使用qt designer设计ui。 位置在C:\Users\a1090\Documents\sitting-posture-correction-python-env\Lib\site-packages
# \qt5_applications\Qt\bin\designer.exe
# import neural network and ui data
try:
    #    from classification23_taichi_eigenvalue import *
    from mainWindow import *
#    from data_process import *
#    from predict_eigenvalue import *
except ImportError as e:
    raise e

# Import Openpose (Windows/Ubuntu/OSX)
# os.path.dirname(os.path.realpath(__file__))
# openPose的位置
openPose_path = r"C:/Users/a1090/Documents/GitHub/openpose"
try:
    # Windows Import
    if sys.platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(rf'{openPose_path}/build/python/openpose/Release')
        os.environ['PATH'] = os.environ[
                                 'PATH'] + ';' + rf'{openPose_path}/build/x64/Release;' + rf'{openPose_path}/build/bin;'
        import pyopenpose as op
    else:
        print(f"Error: not implemented on the {sys.platform} platform, stop importing")
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python '
        'script in the right folder?')
    raise e
finally:
    pass

# args 设置
# 详细参考flags.hpp 文件
params = {
    "model_folder": rf"{openPose_path}\models",  # model位置
    "render_threshold": 0.001,  # 控制骨骼关节点最低置信度
    # "camera_resolution": "180x88",
    "number_people_max": 1,  # 识别人数
    "keypoint_scale": 3  # `3` to scale it in the range [0,1], where (0,0) would be the top-left corner of the
    # image, and (1,1) the bottom-right one
}
# 启动openPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 坐姿姿态数组，顺序与tag相对应
pos = ["无法识别的姿势",
       "正确的坐姿",
       "头部倾斜",
       "身体前倾",
       "身体后倾",
       "身体右倾",
       "身体左倾"
       ]


class Video:
    def __init__(self, capture):
        self.capture = capture
        self.currentFrame = None
        self.previousFrame = None

    def captureFrame(self):
        """
        capture frame and return captured frame
        """
        ret, readFrame = self.capture.read()
        return readFrame

    def captureNextFrame(self):
        """
        capture frame and reverse RBG BGR and return opencv image
        """
        ret, readFrame = self.capture.read()
        if ret:
            self.currentFrame = cv2.cvtColor(readFrame, cv2.COLOR_BGR2RGB)

    def convertFrame(self):
        # converts frame to format suitable for QtGui
        try:
            height, width = self.currentFrame.shape[:2]
            img = QImage(self.currentFrame, width, height, QtGui.QImage.Format_RGB888)
            img = QPixmap.fromImage(img)
            self.previousFrame = self.currentFrame
            return img
        except cv2.Error:
            return None


class mWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mWindow, self).__init__()
        self.setupUi(self)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.showCapture)
        self.video = Video(cv2.VideoCapture(0))
        self._timer.start(100)  # 每隔多长时间


    def showCapture(self):
        try:

            frame = self.video.captureFrame()
            datum = op.Datum()
            datum.cvInputData = frame
            # print(frame)
            print(1)
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            print(2)
            resPic = datum.cvOutputData
            # cv2.putText(resPic, "OpenPose", (25, 25),
            #             cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 222))
            print(2.5)
            pic = cv2.cvtColor(resPic, cv2.COLOR_BGR2RGB)
            pic = QImage(pic, pic.shape[1], pic.shape[0], QtGui.QImage.Format_RGB888)
            self.label_3.setPixmap(QPixmap.fromImage(pic))
            print(3)
            keyPoints = datum.poseKeypoints
            print(3.5)
            if datum.poseKeypoints is None:
                print("no body")
                self.label_4.setText("未识别到人体")
                return
            else:
                print(3.8)
                keyPointsList = [keyPoints[0][i][j] for i in range(25) for j in range(3)]
            print(4)
            resultTag=predictResult(keyPointsList)
            self.label_4.setText(pos[resultTag])
            print(5)
            print(pos[resultTag])
        except TypeError as err:
            print(f"type: {err}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = mWindow()
    mainWindow.setStyleSheet("#MainWindow{border-image:url(../sundry/back5.png);}")
    mainWindow.show()
    sys.exit(app.exec_())
