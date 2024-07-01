import scipy.io
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt

from data import CrackForest

# for index in range(1, 10):
#     mat = scipy.io.loadmat(f"./data/CrackForest-dataset/groundTruth/00{index}.mat")
#
#     np_seg = mat["groundTruth"][0][0][0]
#     t_seg = torch.from_numpy(np_seg)
#     t_seg = torch.where(t_seg > 1, 0, 255)
#
#     cv2.imwrite(f"temp/00{index}.png", t_seg.detach().cpu().numpy())

# numbers = list(range(1, 119))
#
# # Format each number as a string with leading zeros
# string_list = ["{:03d}".format(num) for num in numbers]
#
# print(string_list)

# dm = CrackForest()
# train_dataloader = dm.train_dataloader()
#
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")
