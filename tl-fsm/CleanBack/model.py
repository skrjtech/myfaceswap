import os
import sys
import warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "default"
import cv2
from utils.types import *
from tqdm import tqdm

def aspectRatio(shapeInp: tuple, shapeOut: tuple) -> Tuple[int, int]:

    widthInp, heightInp = shapeInp
    widthOut, heightOut = shapeOut
    aspectInp = widthInp / heightInp
    aspectOut = widthOut / heightOut

    if aspectOut >= aspectInp:
        wOut, hOut = (round(heightOut * aspectInp), heightOut)
    else:
        wOut, hOut = (widthOut, round(widthOut / aspectInp))

    return (wOut, hOut)

class Resize(torchvision.transforms.Lambda):
    def __init__(self, WIDTH: int, HEIGHT: int):
        super(Resize, self).__init__(lambd=lambda x: cv2.resize(x, (WIDTH, HEIGHT)))

class CaptureLoader:
    def __init__(self, camIdx: CaptureT=0, transforms: Transforms=None, batch_size: int=1):
        self.Capture = Cap = cv2.VideoCapture(camIdx)
        self.MAXFRAME = int(Cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.MAXWIDTH = int(Cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.MAXHEIGHT = int(Cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.TRANSFORMS = transforms if transforms is not None else lambda x: x
        self.BATCHSIZE = batch_size
        self.COUNT = 0

        self.LEN = self.MAXFRAME // self.BATCHSIZE
        if self.LEN != 0:
            self.LEN += 1

    def __iter__(self):
        return self

    def __len__(self):
        return self.LEN

    def __getitem__(self, idx: Optional[int] = None):
        if self.Capture.isOpened():
            ret, frame = self.Capture.read()
            if ret:
                return frame, self.TRANSFORMS(frame), idx

    def __next__(self):
        if self.COUNT == self.MAXFRAME:
            raise StopIteration()
        Container = []
        for _ in range(self.BATCHSIZE):
            if self.COUNT == self.MAXFRAME:
                return zip(*Container)
            Container.append(self[self.COUNT])
            self.COUNT += 1
        return zip(*Container)


class Base(object):
    MODEL: TModule = None
    def __init__(
            self,
            width: int = 320,
            height: int = 320,
            camIdx: CaptureT = 0,
            batchSize: int = 1,
            outputPath: str = '',
            device: Device = 'cpu'
    ):
        self.BASEWIDTH = width
        self.BASEHEIGHT = height
        self.CAPINDEX = camIdx
        self.BATCHSIZE = batchSize
        self.OUTPUTPATH = outputPath
        self.DEVICE = device
        self.Build()

    def Build(self, *args, **kwargs):
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        model.eval()
        if self.DEVICE == 'cuda':
            if torch.cuda.is_available():
                self.DEVICE = 'cuda:0'
                model.cuda()
        transform = torchvision.transforms.Compose([
            Resize(320, 320),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        CapData = CaptureLoader(camIdx=self.CAPINDEX, transforms=transform, batch_size=self.BATCHSIZE)
        self.MODEL = model
        self.Cap = CapData

    def Predict(self, *args, **kwargs) -> NPArray:
        return self.MODEL(*args, **kwargs)['out'].argmax(1).byte().cpu().numpy()


    def Masking(self, inputs: Tensor):
        predict = self.Predict(inputs)
        mask = np.where(predict > 0., 1., 0.) * 255.
        mask = mask.astype(np.uint8)
        mask = map(lambda x: np.tile(x, (3, 1, 1)).transpose((1, 2, 0)), mask)
        return mask

def ImageSave(img, path, idx):
    path_ = os.path.join(path, f"{idx:0=10d}.png")
    cv2.imwrite(path_, img)

class VideoFileCleanBackModel(Base):
    def __init__(self, *args, **kwargs):
        super(VideoFileCleanBackModel, self).__init__(*args, **kwargs)

    def Run(self):
        for images, frames, idxs in tqdm(self.Cap):
            inputs = torch.stack(frames)
            msk = self.Masking(inputs.to(self.DEVICE))
            msk = list(map(lambda x: cv2.resize(x, (self.Cap.MAXWIDTH, self.Cap.MAXHEIGHT)), msk))
            out = list(map(lambda x: cv2.bitwise_and(*x), zip(images, msk)))
            for (O, I) in zip(out, idxs):
                ImageSave(O, self.OUTPUTPATH, I)

class PluginCamFrameCleanBackModel(Base):
    def __init__(self, *args, **kwargs):
        super(PluginCamFrameCleanBackModel, self).__init__(*args, **kwargs)

    def Plugin(self) -> NPArray:
        images, frame, _ = self.Cap[-1]
        inputs = torch.Tensor(frame).unsqueeze(0).to(self.DEVICE)
        msk = list(self.Masking(inputs))
        msk = list(map(lambda x: cv2.resize(x, (self.Cap.MAXWIDTH, self.Cap.MAXHEIGHT)), msk))
        out = list(map(lambda x: cv2.bitwise_and(*x), zip([images], msk)))[-1]
        return out

if __name__ == '__main__':
    # 4K
    # print(aspectRatio((3840, 2160), (320, 320))) # -> (320, 180)
    transform = torchvision.transforms.Compose([
        Resize(320, 320),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # cap = CaptureLoader('/DATABASE/TrainVideo/source.mp4', batch_size=4, transforms=transform)
    # for i, (imgs, frames, idxs) in enumerate(cap):
    #     print(i+1, idxs)
    CAP = PluginCamFrameCleanBackModel(camIdx='/DATABASE/TrainVideo/source.mp4', device='cuda')
    while True:
        try:
            frame = CAP.Plugin()
            if cv2.waitKey(1) == ord('q'):
                break
            cv2.imshow("frame", frame)
        except KeyboardInterrupt():
            break
    cv2.destroyAllWindows()