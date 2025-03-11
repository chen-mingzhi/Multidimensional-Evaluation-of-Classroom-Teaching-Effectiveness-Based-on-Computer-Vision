import sys
sys.path.append('E:/code/py/github/GFPGAN/Facial_Expression_Recognition')

import cv2
import torch
from models import VGG
from PIL import Image
import transforms as transforms

# 加载训练好的模型
def load_model(model_path):
    net = VGG('VGG19')
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    net.cuda()
    net.eval()
    return net

# 定义表情类别
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 定义图像预处理函数
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),  # 将图像转换为灰度图，并指定输出通道数为3
    transforms.ToTensor()
])


# 使用 OpenCV 捕获摄像头实时视频流
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 加载模型
model_path = 'Facial_Expression_Recognition/FER2013_VGG19/PrivateTest_model.t7'
net = load_model(model_path)

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # 读取视频流的帧
    ret, frame = cap.read()

    # 将彩色图像转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用人脸检测器检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 在检测到的人脸周围绘制矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 将人脸区域进行预测
        face_img = gray[y:y+h, x:x+w]
        pil_img = Image.fromarray(face_img)
        img = transform(pil_img).unsqueeze(0).cuda()
        outputs = net(img)
        _, predicted = torch.max(outputs.data, 1)
        cv2.putText(frame, class_names[int(predicted)], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # 显示实时视频流
    cv2.imshow('Real-time Facial Expression Detection', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
