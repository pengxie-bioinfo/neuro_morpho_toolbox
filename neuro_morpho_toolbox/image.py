import SimpleITK as sitk
import numpy as np

class image:
    def __init__(self, file):
        self.file = file
        self.image = sitk.ReadImage(file)
        self.array = sitk.GetArrayViewFromImage(self.image)
        self.size = self.image.GetSize()
        self.space = self.image.GetSpacing()

