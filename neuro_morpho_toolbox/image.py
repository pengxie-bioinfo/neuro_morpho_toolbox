import SimpleITK as sitk
import numpy as np

class image:
    def __init__(self, file):
        self.file = file
        self.image = sitk.ReadImage(file)
        self.array = sitk.GetArrayViewFromImage(self.image)
        self.array = np.swapaxes(self.array, 0, 2)
        # self.array = np.flip(self.array, 1)
        self.axes = ["x", "y", "z"]
        self.size = dict(zip(self.axes, self.image.GetSize()))
        self.space = dict(zip(self.axes, self.image.GetSpacing()))
        self.micron_size = dict(zip(self.axes,
                                    [self.size["x"] * self.space["x"],
                                     self.size["y"] * self.space["y"],
                                     self.size["z"] * self.space["z"],
                                     ]
                                    ))
