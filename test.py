import os.path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.nn import functional as F
import torch

from model import UNet
from torchvision import transforms

out_threshold = 0.2

if __name__ == "__main__":
    for i in [1, 101, 117, 302]:
        unet = UNet.load_from_checkpoint(
            # ".\\MFFU-Net\\wj2g2u80\\checkpoints\\epoch=99-step=1300.ckpt"
            # ".\\MFFU-Net\\eznka001\\checkpoints\\epoch=99-step=1300.ckpt"
            os.path.join(
                "lightning_logs", "version_1", "checkpoints", "epoch=99-step=7500.ckpt"
            ),
        ).eval()

        img = cv2.imread(f".\\test\\{i}.png")
        img = transforms.ToTensor()(img).to("cuda").unsqueeze(0)

        output = unet(img)
        # output = F.interpolate(output, scale_factor=1, mode="bilinear")
        output = F.interpolate(output, scale_factor=1, mode="bicubic")
        output = torch.sigmoid(output) > out_threshold

        output = output.to("cpu").detach().numpy()
        output = np.squeeze(output)
        output = np.where(output, 1, 0)
        plt.imshow(output)
        plt.show()
        # plt.imsave(os.path.join("test", f"{i}-ca.png"), np.squeeze(output))
        plt.imsave(os.path.join("test", f"{i}-gan.png"), np.squeeze(output))
