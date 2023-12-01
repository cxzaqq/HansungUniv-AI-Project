import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tkinter import *
import imutils
import cv2
from PIL import Image
from PIL import ImageTk
import numpy as np
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
            nn.Linear(250, 15))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

#######################################################

test_dir = 'test'
test_data = dset.ImageFolder(test_dir, transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((100,100)),
    transforms.ToTensor()
    ])
)
test_loader = DataLoader(test_data)

net = SiameseNetwork()
net.load_state_dict(torch.load(('model16_250+15_10+100_3000+10.pth'), map_location=torch.device('cpu')))
net.eval()

test_list = []
with torch.no_grad():
    for img1, label in test_loader:
        output2 = net.forward_once(img1)
        test_list.append([output2, label])

classes = test_data.classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))
################################################### 디버그
for idx, i in enumerate(test_list):
    print('\n=====================', idx , '=====================')
    # print(i[1].item(), end='\t')
    # print(torch.round(i[0]).numpy())
    k = 0
    for j in test_list:
        if j[1] == k:
            if k:
                print('')
            print(f'{classes[k]:10s}', end='\t')
            k += 1
        print(int(F.pairwise_distance(i[0], j[0]).item()), end='\t')
    
########################################################

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
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces):
                for (x, y, w, h) in faces:
                    img0 = gray[y : y+h, x : x+w]
                    name, color, dist, i = face_detection(img0)
                    t = cv2.getTextSize(f'{name}', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    cv2.rectangle(frame, (x, y-t[1]-5), (x+t[0]+5, y), color, -1)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f'{name}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, f'{i}) {dist:.1f}', (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # 디버그
                    
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
    output1 = net.forward_once(img0)

    name = None
    min_dist = float("inf")

    k, lb = 0, 0                                            # 디버그
    print('\n============================================') #

    with torch.no_grad():
        for i, (output2, label) in enumerate(test_list):
            euclidean_dist = F.pairwise_distance(output1, output2)
            
            if label.item() == k:                           #
                if k:                                       #
                    print('')                               #
                print(f'{classes[k]:10s}', end='\t')        #
                k+=1                                        #
            print(f'{euclidean_dist.item():.1f}', end='\t') #
            
            if euclidean_dist.item() < 80:  # margin
                if euclidean_dist.item() < min_dist:
                    lb=i                                    #
                    min_dist = euclidean_dist.item()
                    name = classes[label.item()]
                    color = colors[label.item()]

    if name is None:
        name = '*UNKNOWN'
        color = (0, 0, 255)

    return name, color, min_dist, lb

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
