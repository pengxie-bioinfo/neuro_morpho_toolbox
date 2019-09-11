import neuro_morpho_toolbox as nmt
import numpy as np
import SimpleITK as sitk

x = nmt.annotation.array
x = np.swapaxes(x, 0, 2)
x[np.where(x!=1009)] = 0
x = np.array(x, dtype='uint8')

region_name = nmt.bs.df.loc[1009, "Abbrevation"]

img = sitk.GetImageFromArray(x)
sitk.WriteImage(img, "./"+region_name+".nrrd")
