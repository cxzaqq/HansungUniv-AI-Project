import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from tkinter import *
import imutils
import cv2
from PIL import Image, ImageTk
import os
import time

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 250),
            nn.ReLU(inplace=True),
            nn.Linear(250, 15)
        )


    def forward(self, input):
        x = self.cnn1(input)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x
    
# 테스트 데이터
test_dir = "test"
test_dataset = ImageFolder(
    root=test_dir,
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((100,100)),
        transforms.ToTensor()
    ])
)
test_loader = DataLoader(test_dataset)

model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("model3.pth", map_location=device))

# test
model.eval()

test_list = []
with torch.no_grad():
    for img2, label in test_loader:
        output2 = model(img2.to(device))
        test_list.append([output2, label])

classes = test_dataset.classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def img_update(frame):
    # 촬영된 이미지를 레이블에 업데이트
    img = imutils.resize(frame, width=WIDTH_CAM)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)
    label1.configure(image=img)
    label1.img = img

def test_camera():
    global cap

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH_CAM)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT_CAM)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(10, 10))
            if len(faces):
                for (x, y, w, h) in faces:
                    img0 = gray[y : y+h, x : x+w]
                    name, color = face_detection(img0)
                    t = cv2.getTextSize(f'{name}', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    cv2.rectangle(frame, (x, y-t[1]-5), (x+t[0]+5, y), color, -1)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f'{name}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    label2.configure(text='Face Found', fg="black")
            else:
                label2.configure(text="Face Not Found", fg="dimgray")

            img_update(frame)  # 이미지 업데이트
            root.update()
            time.sleep(0.5)


def face_detection(img0):  # 얼굴 비교
    img0 = torch.from_numpy(img0).float() / 255.0
    img0 = img0.unsqueeze(0).unsqueeze(0)
    img0 = transforms.Resize((100, 100))(img0)
    output1 = model(img0)

    name = None
    min_dist = float("inf")

    with torch.no_grad():
        for i, (output2, label) in enumerate(test_list):
            euclidean_dist = F.pairwise_distance(output1, output2)

            if euclidean_dist.item() < 80:  # margin
                if euclidean_dist.item() < min_dist:
                    min_dist = euclidean_dist.item()
                    name = classes[label.item()]
                    color = colors[label.item()]

    if name is None:
        name = '*UNKNOWN'
        color = (0, 0, 255)

    return name, color

def exit():
    try:
        cap.release()
    except:
        pass
    root.destroy()


DELAY_CAM = 100
DELAY_END = 2000
WIDTH = 800
HEIGHT = 450
WIDTH_CAM = 400
HEIGHT_CAM = 300

# UI 초기화
root = Tk()
root.protocol("WM_DELETE_WINDOW", exit)
root.minsize(WIDTH, HEIGHT)

label1 = Label(root, bitmap="hourglass", bd=3, width=WIDTH_CAM, height=HEIGHT_CAM, relief="ridge")
label1.place(relx=0.5, rely=0.35, anchor=CENTER)
label2 = Label(root, bd=2, font=("Calibri",30))
label2.place(relx=0.5, rely=0.8, anchor=CENTER)
button1 = Button(root, text="Exit", height=2, font=("Calibri", 20), command=exit)
button1.place(relx=0.8, rely=0.9, anchor=CENTER)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
root.after(DELAY_CAM*5, test_camera)

root.mainloop()