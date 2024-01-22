import cv2
import os

import numpy as np
import torch
from ultralytics import YOLO
import requests
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(rf'human.pt')

deep_sort_weights = r'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

cam_credentials ={
    'ip' : '10.71.172.253',
    'user' : 'admin',
    'pass': 'admin123'
}


def get_camera_url(ip: str, user: str, password: str):
    return f'rtsp://{user}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype=1'


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(get_camera_url('10.71.172.253', 'admin', 'admin123'))


def detect():
    global cap

    class_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'truck', 'boat']
    frames = []
    unique_track_ids = set()

    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = model(frame, device=device, classes=0, conf=0.75, iou=0.5)

        for pred in preds:
            img = pred.plot()
            boxes = pred.boxes          # Boxes for bbox outputs
            probs = pred.probs          # Class probabilities for classification outputs
            cls = boxes.cls.tolist()    # convert tensor to list
            xyxy = boxes.xyxy
            conf = boxes.conf
            xywh = boxes.xywh           # boxes with x,y,w,h format (N, 4)
            ids = pred.boxes.id
            for class_index in cls:
                class_name = class_name[int(class_index)]

        pred_cls = np.array(cls)
        conf = conf.detach().cpu().numpy()
        xyxy = xyxy.detach().cpu().numpy()
        bboxes_xywh = xywh.cpu().numpy()

        tracks = tracker.update(bboxes_xywh, conf, og_frame)

        for track in tracker.tracker.tracks:
            track_id = track.track_id
            hits = track.hits
            x1, y1, x2, y2 = track.to_tlbr()
            w = x2 - x1     # Calculate width
            h = y2 - y1     # Calculate Height

            # Setting up color values (B, G, R)
            red_color = (0, 0, 255)
            green_color = (0, 255, 0)
            blue_color = (255, 0, 0)

            color_id = track_id % 3
            match color_id:
                case 0:
                    color = red_color
                case 1:
                    color = green_color
                case _:
                    color = blue_color
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y2 + h)), color, 2)
            text_color = (0, 0, 0)
            cv2.putText(frame, f"{class_name}{track_id}", (int(x1) + 5, int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)
            unique_track_ids.add(track_id)

        cv2.imshow('detect', frame)
        key = cv2.waitKey(1) & 0xff

        if key == ord('q') or key == 27:
            break


if __name__ == '__main__':
    detect()
    cap.release()
    cv2.destroyAllWindows()
