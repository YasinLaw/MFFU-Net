import os

import scipy.io
import numpy as np
import cv2

if __name__ == "__main__":
    # convert ground truth from mat to png image
    for i in range(1, 119):
        mat = scipy.io.loadmat(
            os.path.join(
                "data", "CrackForest-dataset", "groundTruth" + "{:03d}.mat".format(i)
            )
        )
        np_seg = mat["groundTruth"][0][0][0]
        (y, x) = np.where(np_seg == 2)
        np_seg[y, x] = 255
        (y, x) = np.where(np_seg == 1)
        np_seg[y, x] = 0

        cv2.imwrite(
            os.path.join("data", "CrackForest-dataset", "gt", f"{i}.png"), np_seg
        )

    for i in range(1, 119):
        # convert image from jpg to png
        img = cv2.imread(
            os.path.join("data", "CrackForest-dataset", "image", "{:03d}.jpg".format(i))
        )

        cv2.imwrite(os.path.join("data", "CrackForest-dataset", "img", f"{i}.png"), img)
