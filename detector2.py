import cv2

import numpy as np
import torch
from ultralytics import YOLO
import cvzone
from deep_sort.deep_sort import DeepSort

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(rf'/home/joker/coding/ppe/vivek chettan/ppe/human.pt')

deep_sort_weights = r'/home/joker/coding/ppe/vivek chettan/ppe/deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=25)

cap = cv2.VideoCapture(0)


# def detect(frame):
#     global width, height

#     og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = YOLO(rf'yolov8s.pt') 

#     preds = model(frame, device=device, classes=0, conf=0.75, iou=0.5)

   
#     for pred in preds:
#         boxes=pred.boxes.xyxy
#         xywh=pred.boxes.xywh
#         conf = pred.boxes.conf #.cpu().detach().numpy()
#     conf = conf.detach().cpu().numpy()
#     bboxes_xywh = xywh.cpu().numpy()
    
#     trackers = track(bboxes_xywh, conf, og_frame)  


def detect():
    global device,cap

    trackers=[]

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    human_model = YOLO(rf'yolov8s.pt')
    ppe_model = YOLO(rf'ppe_sneha.pt')

    while True:
        ret,frame = cap.read()
        key = cv2.waitKey(1) & 0xff
        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        preds = human_model(frame, device=device, classes=0, conf=0.3, iou=0.5)

        for pred in preds:
            
            boxes=pred.boxes.xyxy
            boxes = boxes.cpu().detach().numpy()

            for box in boxes:
                x1,y1,x2,y2=box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                cropped=frame[y1:y2, x1:x2]

                results=ppe_model(cropped, device=device, classes=[2,3,4], conf=0.5, iou=0.5)

                for result in results:
                    ppe_boxes=result.boxes.xyxy
                    xywh=result.boxes.xywh
                    conf = result.boxes.conf #.cpu().detach().numpy()
                    conf = conf.detach().cpu().numpy()
                    bboxes_xywh = xywh.cpu().numpy()
                    class_index=[]
                    for box in result.boxes:
                        index = int(box.cls)
                        class_index.append(index)
                        
                        count=len(class_index)
                        if count>0:
                            trackers = track(bboxes_xywh, conf, og_frame)
                            for tracker in trackers:
                                x1, y1, x2, y2, w, h = tracker['bbox'][0], tracker['bbox'][1], tracker['bbox'][2],tracker['bbox'][3], tracker['bbox'][4], tracker['bbox'][5]
                                id = tracker['track_id']  

                                text_color = (255, 255, 255)
                                # Setting up color values (B, G, R)
                                red_color = (0, 0, 255)
                                green_color = (0, 255, 0)
                                blue_color = (255, 0, 0)

                                color_id = id % 3
                                match color_id:
                                    case 0:
                                        color = red_color
                                    case 1:
                                        color = green_color
                                    case _:
                                        color = blue_color

                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y2 + h)), color, 2)
                                cvzone.putTextRect(frame, f'Person {id}', (int(x1) + 1, int(y1) - 10), 1, 1, text_color, color, cv2.FONT_HERSHEY_SIMPLEX, offset=3, border=1, colorB=color)
                                # unique_track_ids.add(track_id)

                                cv2.imshow('frame', frame)
                                


def track(bboxes_xywh, conf, og_frame):

    tracks = tracker.update(bboxes_xywh, conf, og_frame)
    data = []
    for track in tracker.tracker.tracks:
        t_dict = {}
        t_dict['track_id'] = track.track_id
        x1, y1, x2, y2 = track.to_tlbr()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w = x2 - x1     # Calculate width
        h = y2 - y1     # Calculate Height
        t_dict['bbox'] = [x1, y1, x2, y2, w, h]
        # Setting up color values (B, G, R)
        data.append(t_dict)
    
    return data


if __name__ == '__main__':
    detect()
    cap.release()
    cv2.destroyAllWindows()
