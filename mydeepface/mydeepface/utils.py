from email.mime import image
import torch
import numpy as np
from PIL import Image

def tensor2image(tensor: torch.tensor) -> np.asarray:
    image = 127.5 * (tensor[0].cpu().float().detach().numpy() + 1.0)
    if image.shape[0] == 1: image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)

def tensor2image_ver2(tensor: torch.tensor) -> np.asarray:
    image = 127.5 * (tensor[0].cpu().float().detach().numpy() + 1.0)
    if image.shape[0] == 1: image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8).transpose(1, 2, 0)

def save_img(img_real: np.asarray, img_fake: np.asarray, img_rec: np.asarray, save2path: str):
    Image.fromarray(np.concatenate((img_real, img_fake, img_rec), axis=1)).save(save2path)

def save_saveral_img(imgs: tuple, save2path: str):
    Image.fromarray(np.concatenate(imgs, axis=1)).save(save2path)
