import imaplib
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms

cap = cv2.VideoCapture(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(320,transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if ret:
            W, H, C = frame.shape
            cv2.imshow("frame", frame)
            flip_img = cv2.flip(frame, 1)
            cv2.imshow("flip 1 > 0", flip_img)
            input_tensor = preprocess(flip_img)
            input_batch = input_tensor.unsqueeze(0).to(device)
            output = model(input_batch)['out'][0].argmax(0).byte().cpu().numpy()
            output[output > 0] = 255
            output[output != 255] = 0
            mask = cv2.resize(output, (H, W)).reshape(W, H, 1)
            mask = np.concatenate([mask, mask, mask], axis=-1)
            output = cv2.bitwise_and(flip_img, mask)
            cv2.imshow('mask', mask)
            cv2.imshow('output', output) 
            if cv2.waitKey(1) == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()