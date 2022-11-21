import sys
import cv2
import numpy as np
import torch
import torchvision

class Base(object):
    def __init__(self, camera: str or int, imageSize: tuple=(640, 480), gpu: bool=False):
        self.CapVideo = cv2.VideoCapture(camera)
        self.imageSize = imageSize
        self.gpu = gpu
        self.device = 'cpu'
        # Camera Info
        self.MAXHEIGHT = int(self.CapVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.MAXWIDTH = int(self.CapVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.FPS = int(self.CapVideo.get(cv2.CAP_PROP_FPS))

    def normalView(self, AutoControll: bool=True, windowName: str='OriginalFrame', exitCommand: str='q'):
        # Camera Controll All Off
        if not AutoControll:
            self.CapVideo.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self.CapVideo.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.CapVideo.set(cv2.CAP_PROP_AUTO_WB, 0)
        while self.CapVideo.isOpened():
            ret, frame = self.CapVideo.read()
            if not ret: break
            cv2.imshow(windowName, frame)
            if cv2.waitKey(1) == ord(exitCommand): break
        cv2.destroyWindow(windowName)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.CapVideo.release()

class RealTimePreprocess(Base):
    def replace(self, CaptureName: str="OutputFrame", exitCommand: str='q', AutoControll: bool=True):
        # Camera Controll All Off
        if not AutoControll:
            self.CapVideo.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self.CapVideo.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.CapVideo.set(cv2.CAP_PROP_AUTO_WB, 0)
        while self.CapVideo.isOpened():
            ret, frame = self.CapVideo.read()
            backgraund = np.full_like(frame, 255).astype(np.float32)
            if not ret: break
            resize = cv2.resize(frame, (320, 320)) # frame: Height, Width, Channel -> 320, 320, Channel
            inp = self.transforms(resize).unsqueeze(0).to(self.device) # inp: 320, 320, Channel -> Batch, Channel 320, 320
            output = self.model(inp)['out'].squeeze(0) # output: Batch, Channel, 320, 320 -> Channel, 320, 320
            mask = output.squeeze(0).argmax(0).byte().cpu().numpy() #
            mask = (np.where(mask > 0, 1, 0) * 255.).astype(np.uint8)
            mask = np.tile(mask, (3, 1, 1)).transpose((1, 2, 0))
            output = cv2.bitwise_and(resize, mask)
            cv2.imshow(CaptureName, output)
            if cv2.waitKey(1) == ord(exitCommand): break
        cv2.destroyWindow(CaptureName)

class RealTimeProcess(RealTimePreprocess):
    def __init__(self, *args, **kwargs):
        super(RealTimeProcess, self).__init__(*args, **kwargs)
        self.modelBuild()

    def modelBuild(self):
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.eval()
        if self.gpu:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                self.model.cuda()

    def recode(self, path: str='output.mp4', AutoControll: bool=True, windowName: str='OriginalFrame', exitCommand: str='q'):
        # Camera Controll All Off
        if not AutoControll:
            self.CapVideo.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self.CapVideo.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.CapVideo.set(cv2.CAP_PROP_AUTO_WB, 0)
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(path, codec, self.FPS, (self.MAXWIDTH, self.MAXHEIGHT))
        while self.CapVideo.isOpened():
            ret, frame = self.CapVideo.read()
            if not ret: break
            video.write(frame)
            cv2.imshow(windowName, frame)
            if cv2.waitKey(1) == ord(exitCommand): break
        video.release()
        cv2.destroyWindow(windowName)

if __name__ == '__main__':
    # RealTimeProcess(0, gpu=True).replace(AutoControll=False)
    RealTimeProcess(0, gpu=True).recode(AutoControll=False)