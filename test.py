import cv2
import numpy as np
from matplotlib import pyplot as plt

from model import UNet
from torchvision import transforms

if __name__ == "__main__":
    unet = UNet.load_from_checkpoint(
        ".\\MFFU-Net\\wj2g2u80\\checkpoints\\epoch=99-step=1300.ckpt"
    ).eval()

    img = cv2.imread(".\\test\\1.png")
    img = transforms.ToTensor()(img).to("cuda").unsqueeze(0)

    output = unet(img)
    output = output.squeeze(0)
    output = output.to("cpu").detach().numpy()

    plt.imshow(np.squeeze(output))
    plt.show()
