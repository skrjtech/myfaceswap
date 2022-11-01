import cv2
import numpy as np

import torch
import torchvision
from torchvision import transforms

cap = cv2.VideoCapture("base1.mp4")
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
            H, W, C = frame.shape
            flip_img = cv2.flip(frame, 1)
            input_tensor = preprocess(flip_img)
            input_batch = input_tensor.unsqueeze(0).to(device)
            input_batch = torch.cat((input_batch, input_batch, input_batch))
            output = model(input_batch)['out'] #.argmax(0).byte().cpu().numpy()
            output_ = []
            for d in output:
                output_.append(
                    d.argmax(0).unsqueeze(0).byte().cpu().numpy()
                )
            output = np.concatenate(output_)
            print(input_batch.shape, output.shape)
            output = np.where(output > 0, 255, 0)
            mask = cv2.resize(output.astype(np.uint8), (W, H))
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            output = cv2.bitwise_and(flip_img, mask)
            cv2.imshow("frame", frame)
            cv2.imshow("flip 1 > 0", flip_img)
            # cv2.imshow('mask', mask)
            cv2.imshow('output', output) 
            if cv2.waitKey(1) == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()