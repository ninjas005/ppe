import cv2
import os
import torch
from ultralytics import YOLO
import requests

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(rf'human.pt')

cap = cv2.VideoCapture(0)


def detect():
    global cap
    while True:
        _, frame = cap.read()
        preds = model(frame, device=device, classes=0, conf=0.75, iou=0.5)

        for pred in preds:
            img = pred.plot()
            boxes = pred.boxex.xyxy
            ids = pred.boxes.id


        cv2.imshow('detect', img)
        key = cv2.waitKey(1) & 0xff

        if key == ord('q') or key == 27:
            break


if __name__ == '__main__':
    detect()
    cap.release()
    cv2.destroyAllWindows()


#
# import torch
# import cv2
# import numpy as np
# from ultralytics import YOLO
#
# device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = YOLO(rf'human.pt')
#
# cap = cv2.VideoCapture(0)
#
#
# while True:
#     ret,frame = cap.read()
#     preds = model(frame, device=device, classes=0, conf=0.75, iou=0.5)
#
#     for pred in preds:
#         img = pred.plot()
#         boxes=pred.boxes.xyxy
#         boxes = boxes.cpu().detach().numpy()
#         ids=pred.boxes.id
#         print(f'{boxes}\t{ids}')
#         # print (f'1. {boxes[0]}\n {boxes.xyxy.cpu.detach()numpy()}')
#
#     cv2.imshow('Human', np.array(img,dtype = np.uint8))
#     if cv2.waitKey(1) & 0xff == ord('q'):
#     	break
#
# cap.release()