# @Time     : 2018/10/19
# @Author   : Haldate
# 此文件用于读取分类器，并进行预测评估测试

import os
import cv2 as cv
import numpy as np
#import matplotlib.pyplot as plt

# 设置分类器文件路径
recognizer_Path = "C:/Users/lxdn/PycharmProjects/untitled/"
cascPath = "F:/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"

faceCascade = cv.CascadeClassifier(cascPath)
personNames = np.load("C:/Users/lxdn/PycharmProjects/untitled/nameList.npy")


def getPersonName(id, conf):
    return personNames[id]

recognizer = "MyFacePCAModel.xml"
recognizer1="MyFaceLDAModel.xml"

classifier = cv.face.EigenFaceRecognizer_create()
classifier.read(recognizer_Path + recognizer)
classifier1=cv.face.FisherFaceRecognizer_create()
classifier1.read(recognizer_Path + recognizer1)

# 调用摄像头
camera = cv.VideoCapture(0)
# 设置文字
# font = cv.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
font = cv.FONT_HERSHEY_SIMPLEX

while True:
    # 从中读取图像
    rect, image = camera.read()
    image = cv.flip(image,1)
    # 将读取出的图像转化为灰度图
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.3,5)

    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        videoGray = cv.resize(gray[y:y + h,x:x + w], (200, 200))
        imageId, conf = classifier.predict(videoGray)
        imageId1,conf1=classifier1.predict(videoGray)
        personName = getPersonName(imageId, conf)
        personName1=getPersonName(imageId1,conf1)
        print("Eigenfaces")
        print(imageId,conf)
        print(personName)
        print("FisherFace")
        print(imageId1,conf1)
        print(personName1)
        cv.putText(image, personName, (x,y-20),font, 1, (0, 255, 0), 2)
        cv.putText(image, personName1, (x, y-80), font, 1, (255, 0, 0), 2)

    cv.imshow('camera', image)

    if cv.waitKey(10) & 0xff == ord('q'):
        break

camera.release()
cv.destroyAllWindows()
print("Test Done")