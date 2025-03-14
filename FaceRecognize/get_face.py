import sys
sys.path.append('E:/code/py/github/GFPGAN/FaceRecognize')

import os
import random
import cv2
import torch
from facenet_pytorch import MTCNN

# 选择设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


def apply_data_augmentation(image):
    # 水平反转
    if random.random() <= 0.5:
        image = cv2.flip(image, 1)

    # 随机对比度变化
    contrast_factor = random.uniform(0.5, 1.5)
    image = cv2.convertScaleAbs(image, alpha=contrast_factor)

    # 随机亮度变化
    brightness_delta = random.randint(-50, 50)
    image = cv2.add(image, brightness_delta)

    # 随机灰度变换
    if random.random() > 0.5:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def get_face(name, num):
    if not os.path.exists(f'faces/{name}'):
        os.makedirs(f'faces/{name}')
    # mtcnn检测人脸位置
    mtcnn = MTCNN(device=device, keep_all=True)

    # 初始化视频窗口
    windows_name = 'face'
    cv2.namedWindow(windows_name)
    cap = cv2.VideoCapture(0)

    count = 0
    while True:
        # 从摄像头读取一帧图像
        success, image = cap.read()
        if not success:
            break

        # 检测人脸位置
        boxes, probs = mtcnn.detect(image)
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                # 设置人脸检测阈值
                if prob < 0.9:
                    continue

                x1, y1, x2, y2 = [int(p) for p in box]

                # 将当前人脸保存为图片
                face_image_name = f'faces/{name}/{name}_{count}.jpg'
                augmentation_image_name = f"faces/{name}/augmentation_{name}_{count}.jpg"
                count += 1
                if count > num:
                    break
                print(face_image_name)
                face_image = image[y1 - 10:y2 + 10, x1 - 10: x2 + 10]

                augmented_image = apply_data_augmentation(face_image)
                cv2.imwrite(augmentation_image_name, augmented_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                cv2.imwrite(face_image_name, face_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

                # 框出人脸位置
                cv2.rectangle(image, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), color=(0, 255, 0), thickness=2)
                cv2.putText(image, f'num:{count}', (x1, y1 - 30), cv2.FONT_ITALIC, 1, (255, 0, 255), 4)

        # 显示处理后的图片
        cv2.imshow(windows_name, image)

        if count > num:
            break
        # 保持窗口
        key = cv2.waitKey(1)
        if key & 0xff == 27:
            break

    # 释放设备资源，销毁窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    name = input('请输入这个人的名字：')
    get_face(name, 10)
    print('人脸录入完成')
