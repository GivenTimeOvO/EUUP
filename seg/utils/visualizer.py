import torch
from torchvision.transforms.functional import to_pil_image, resize, InterpolationMode
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
import datetime

from fcore.dataset.transforms import Denormalize
from fcore.utils.structure import GenericData


class Visualizer:
    def __init__(self):
        pass

    def __call__(self, img, label, pred=None, cmap=None, label_tensor=None):
        label = resize(label.unsqueeze(0), img.shape[1:], InterpolationMode.NEAREST).squeeze(0)
        if pred is not None:
            pred = resize(pred.unsqueeze(0), img.shape[1:], InterpolationMode.NEAREST).squeeze(0)
            pred = pred.expand_as(label)

        img = img.detach().cpu().numpy()
        label = label.detach()
        label = torch.where(label_tensor == 1, label, 0).cpu().numpy()
        label[label == 255] = 21

        if pred is not None:
            pred = pred.detach()
            pred = torch.where(label_tensor.squeeze(0) == 1, pred, 0).cpu().numpy()

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        if cmap is None:
            cmap = np.zeros((22, 3), dtype=np.uint8)
            for i in range(21):
                r = g = b = 0
                c = i
                for j in range(8):
                    r = r | (bitget(c, 0) << 7 - j)
                    g = g | (bitget(c, 1) << 7 - j)
                    b = b | (bitget(c, 2) << 7 - j)
                    c = c >> 3

                cmap[i] = np.array([r, g, b])
            cmap[21] = np.array(([255, 255, 255]))
        label = cmap[label]
        if pred is not None:
            pred = cmap[pred]

        if label.ndim == 4:
            label = label[0]
        if pred is not None and pred.ndim == 4:
            pred = pred[0]

        if pred is not None:
            img = np.concatenate((img, label.transpose((2, 0, 1)), pred.transpose((2, 0, 1))), axis=1).transpose(
                (1, 2, 0)
            )
        else:
            img = np.concatenate((img, label.transpose((2, 0, 1))), axis=2).transpose((1, 2, 0))
        img = to_pil_image(img)

        return img

def vis(data_list, result_list=None, out_path=None, cmap=None):

    visualizer = Visualizer()

    if isinstance(data_list, GenericData):
        data_list = [data_list]
        result_list = [result_list]
    elif isinstance(data_list, list):
        pass
    else:
        raise ValueError("data_list need to be a GenericData object or list of GenericData.")

    if result_list is None:
        result_list = [result_list] * len(data_list)

    imgs = []
    denorm = Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True, False)
    for i, (data, result) in enumerate(zip(data_list, result_list)):
        data = denorm(data)
        img = data.img.cpu().to(torch.uint8)
        label = data.label.cpu()
        cur_size = torch.ones(data.meta.cur_size).unsqueeze(0)
        label_tensor = torch.zeros(1, 512, 512)
        label_tensor[:, :cur_size.shape[1], :cur_size.shape[2]] = cur_size
        if result is not None:
            result = result.cpu()
        img = visualizer(img, label, result, cmap, label_tensor)
        img = np.asarray(img)
        imgs.append(img)

    imgs = np.concatenate(imgs, axis=0)

    plt.figure(figsize=(imgs.shape[1] / 200, imgs.shape[0] / 200), dpi=200)
    plt.subplots_adjust(bottom=0, top=1, left=0, right=1)
    plt.imshow(imgs)

    if out_path is None:
        plt.show()
    else:
        cur_time = datetime.datetime.now()
        time = cur_time.timestamp()
        filename = str(int(time)) + ".jpg"

        plt.savefig(out_path + filename)
        # plt.show()




        