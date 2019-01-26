import cv2
import numpy as np  # 添加模块和矩阵模块
# 图片保存路径
save_face_path = 'F:/photo_saving/LWL/2016317200424'
save_gray_path = 'F:/gray_train_set/LWL/2016317200424'  # 保存照片的文件夹地址
# 获取人脸识别特征数据
face_cascade = cv2.CascadeClassifier("F:/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("F:/Python37-32/Lib/site-packages/cv2/data/haarcascade_eye.xml")
cap = cv2.VideoCapture(0)  # VideoCapture类用于处理摄像头/视频读取_写入操作。
# 0代表0号摄像头
# 打开摄像头，若打开本地视频，同opencv一样，只需将０换成("×××.avi")
i = 1
while(1):    # 持续不断的get a frame
    ret, frame = cap.read()    # show a frame 第一个参数ret 为True 或者False,代表有没有读取到图片,第二个参数frame表示截取到一帧的图片
    # 镜像翻转
    frame = cv2.flip(frame, 1)
    # 灰度转换，gray就是灰度图片
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # BGR to gray
    # 对于BGR，blue在高位，green在中位，red在低位，正好与RGB相反
    # 探测图片中的人脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 会得到一串list:size(人脸个数)，[x,y,h,w](人脸的位置)
    print("faces", faces)  # 输出每个检测到的脸的list
    # 检测眼睛
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    if not faces is ():  # 如果faces不为空
        for x, y, z, w in faces:  # 同时在faces这个数组中迭代4个参数x,y,z,w。x,y是矩阵左上点的坐标，z是矩阵的宽，w是矩阵的高
            roiImg = frame[y:y+w, x:x+z]  # ROI，即感兴趣区域，用roiImg设置感兴趣区域的图像
            # Python中ROI区域的设置是使用Numpy中的索引来实现的，参数为y的坐标区间和x的坐标区间，w和z为偏移量，保存在roiImg矩阵中
            # 保存人脸图片
            cv2.imwrite(save_face_path+'('+str(i)+')'+'.jpg', roiImg)
            # 保存灰度图片
            roi_gray = cv2.cvtColor(roiImg, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(save_gray_path + '(' + str(i) + ')' + '.jpg', roi_gray)
            # 将人脸用矩形框出来
            cv2.rectangle(frame, (x, y), (x+z, y+w), (0, 255, 0), 2)  # (x,y)为起始坐标,(x+z,y+w)为结束坐标，（0,255,0）是画线对应的rgb颜色，2是画线的宽度
            i += 1  # 用于给保存的图片取编号

    if not eyes is ():  # 如果eyes不为空。在python中None,False,空字符串""，空列表[],空字典{},空元祖(),都相当于false
        for x, y, z, w in eyes:
            # 将眼睛框出来
            cv2.rectangle(frame, (x, y), (x+z, y+w), (255, 0, 0), 2)
            i += 1

    cv2.imshow("capture", frame)  # 以窗口形式显示frame帧 ，窗口名为capture

    if cv2.waitKey(1) & 0xFF == ord(chr(27)):  # 画面延时1ms并和11111111做与运算，若等于关闭键，则break
        break   # waitKey用于设置显示图像的频率，每1ms刷新一次
                # chr(27)为ASCⅡ为27的字符，即ESC，ord函数是将字符转化为ASCⅡ，由于直接输入ESC不是一个字符，因此需要这样转换一下
                # 也可以按q关闭，则写成 if cv2.waitKey(1) & 0xFF == ord('q'): 但这样就对大小写有局限性
#model = cv2.face.LBPHFaceRecognizer_create()
#model=cv2.face.EigenFaceRecognizer_create(	)
#model.train(roiImg, labels)
#model.save("MyFacePCAModel.xml")
cap.release()   # 关闭视频流文件
cv2.destroyAllWindows()  # 释放并销毁窗口
# 识别的时候，不需要再保存图片了，只需要匹配就行