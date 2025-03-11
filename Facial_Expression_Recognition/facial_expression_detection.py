# facial_expression_detection.py

import sys
sys.path.append('E:/code/py/github/GFPGAN/Facial_Expression_Recognition')

import os
import cv2
import torch
from models import VGG
from PIL import Image
import transforms as transforms

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


class FacialExpressionDetector:
    def __init__(self, model_path, image_folder_path, output_folder_path):
        self.net = self.load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.image_folder_path = image_folder_path
        self.output_folder_path = output_folder_path
        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def load_model(self, model_path):
        net = VGG('VGG19')
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        net.cuda()
        net.eval()
        return net

    def process_images(self):
            image_files = [f for f in os.listdir(self.image_folder_path) if os.path.isfile(os.path.join(self.image_folder_path, f))]
            os.makedirs(self.output_folder_path, exist_ok=True)

            # 创建一个txt文件用于存储预测结果
            result_file = open(os.path.join(self.output_folder_path, 'results.txt'), 'w')

            for image_file in image_files:
                image_path = os.path.join(self.image_folder_path, image_file)
                frame = cv2.imread(image_path)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    face_img = gray[y:y+h, x:x+w]
                    pil_img = Image.fromarray(face_img)
                    img = transforms.Compose([
                        transforms.Resize((48, 48)),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor()
                    ])(pil_img).unsqueeze(0).cuda()
                    outputs = self.net(img)
                    _, predicted = torch.max(outputs.data, 1)
                    expression_label = self.class_names[int(predicted)]

                    output_file_path = os.path.join(self.output_folder_path, f"{image_file.split('.')[0]}_{expression_label}.jpg")

                    print(f"Expression Label for {image_file}: {expression_label}")

                    # 将预测结果写入txt文件
                    result_file.write(f"{image_file}: {expression_label}\n")

                    cv2.putText(frame, expression_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    cv2.imwrite(output_file_path, frame)

            # 关闭文件
            result_file.close()

