import cv2, dlib
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()

    if status:
        cv2.imshow("Test", frame)

        # 's' 키가 눌리면 프레임을 캡처하고 저장
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite("registedImg/owner.jpg", frame)
            print("'registedImg/owner.jpg'로 프레임이 캡처되어 저장되었습니다.")
            break

webcam.release()
cv2.destroyAllWindows()

def face_rec(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    
    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    for k,d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)

        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)
        
            shapes.append(shape)

        return rects, shapes, shapes_np

def face_enc(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)

img_paths = {
    'owner': 'registedImg/owner.jpg'
}

descs = {
    'owner': None
}

for name, img_path in img_paths.items():
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    _, img_shapes, _ = face_rec(img_rgb)
    descs[name] = face_enc(img_rgb, img_shapes)[0]

    np.save('registedImg/descs.npy', descs)
    print(descs)