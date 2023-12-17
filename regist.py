from tkinter import *
import imutils
import cv2
from PIL import Image, ImageTk
import os

def img_update(frame):
    # 촬영된 이미지를 레이블에 업데이트
    img = imutils.resize(frame, width=WIDTH_CAM)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)
    label1.configure(image=img)
    label1.img = img


def camera():
    # 카메라 루프
    global cap, crop

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH_CAM)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT_CAM)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces):
                if len(faces) > 1:
                    area_rect = 0
                    for r in faces:
                        area = r[2]*r[3]
                        if area_rect < area:
                            area_rect = area
                            face = r
                else:
                    face = faces[0]

                x, y, w, h = face[0], face[1], face[2], face[3]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                crop = gray[y : y+h, x : x+w]
                btn_config(SETTING)
            else:
                btn_config(NOTFOUND)  # 얼굴이 인식되지 않으면 캡처버튼 누를 수 없음

            img_update(frame)  # 이미지 업데이트
            root.update()


def capture():
    cap.release()
    btn_config(CONFIRM)

def yes():
    global index

    direc = os.path.join('test', textbox.get().replace(' ', '_'))
    if not os.path.exists(direc):
        os.makedirs(direc)
    image_path = os.path.join(direc, f'img_{index}.png')
    cv2.imwrite(image_path, crop)

    index += 1
    if index == 4:
        # 4번의 캡처 완료한 경우
        root.after(DELAY_END, exit)  # 카메라 루프 종료
        btn_config(FINISH)
    else:
        # 다음 단계의 설정 진행
        root.after(DELAY_CAM, camera)
        btn_config(SETTING)

def no():
    btn_config(SETTING)
    camera()

def btn_config(i):
    # 프레임 안의 버튼과 레이블의 설정값을 변경함
    if len(textbox.get())==0:
        label2.configure(text="Enter Your Name First", fg="brown")
        button1.configure(state=DISABLED)
        button2.configure(state=DISABLED)
        button3.configure(state=DISABLED)
    elif i == SETTING:
        label2.configure(text="Face Found", fg="black")
        button1.configure(state=DISABLED)
        button2.configure(state=DISABLED)
        button3.configure(state=NORMAL)
    elif i == CONFIRM:
        label2.configure(text="Confirm?")
        button1.configure(state=NORMAL)
        button2.configure(state=NORMAL)
        button3.configure(state=DISABLED)
    elif i == FINISH:
        label2.configure(text='Setting Finish!')
        button1.configure(state=DISABLED)
        button2.configure(state=DISABLED)
    elif i == NOTFOUND:
        label2.configure(text="Face Not Found", fg="dimgray")
        button3.configure(state=DISABLED)

def exit():
    try:
        cap.release()
    except:
        pass
    root.destroy()

NOTFOUND = -1
SETTING = 0
CONFIRM = 1
FINISH = 2

DELAY_CAM = 100
DELAY_END = 2000
WIDTH = 800
HEIGHT = 600
WIDTH_CAM = 400
HEIGHT_CAM = 300

# UI 초기화
root = Tk()
root.protocol("WM_DELETE_WINDOW", exit)
root.minsize(WIDTH, HEIGHT)

button1 = Button(root, text="Yes", height=2, width=5, fg="crimson", activeforeground="crimson", font=("Calibri",20), command=yes)
button1.place(relx=0.3, rely=0.9, anchor=CENTER)
button2 = Button(root, text="No", height=2, width=5, fg="darkblue", activeforeground="darkblue", font=("Calibri",20), command=no)
button2.place(relx=0.7, rely=0.9, anchor=CENTER)
button3 = Button(root, text="Capture", height=2, width=9, font=("Calibri",15), command= capture)
button3.place(relx=0.5, rely=0.88, anchor=CENTER)
button4 = Button(root, text="Exit", height=2, font=("Calibri",20), command=exit)
button4.place(relx=0.9, rely=0.9, anchor=CENTER)

label1 = Label(root, bitmap='hourglass', bd=3, width=WIDTH_CAM, height=HEIGHT_CAM, relief="ridge")
label1.place(relx=0.5, rely=0.40, anchor=CENTER)
label2 = Label(root, font=("Calibri",20,'bold'))
label2.place(relx=0.5, rely=0.75, anchor=CENTER)

label3 = Label(root, text='Enter Your Name:')
label3.place(relx=0.4, rely=0.08, anchor=CENTER)
textbox = Entry(root)
textbox.place(relx=0.6, rely=0.08, anchor=CENTER)
textbox.focus()

index = 0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
btn_config(0)
root.after(DELAY_CAM*5, camera)

root.mainloop()