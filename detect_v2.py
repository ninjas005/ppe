import torch
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('ppe_sneha.pt')  
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cap = cv2.VideoCapture(0)


while True:
    ret,frame = cap.read()
    
    preds=model(frame, device=device, classes=[2,3,4], conf=0.3, iou=0.5)
    # predictions = model.predict(frame, device=0, classes=0, save_crop=True)

    for pred in preds:
        img = pred.plot()
    cv2.imshow('PPE', np.array(img,dtype = np.uint8))
    if cv2.waitKey(1) & 0xff == ord('q'):
    	break
cap.release()
