import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# Image process
import SimpleITK as sitk

from .swc import swc
from .image import image
from .brain_structure import brain_structure
from .neuron_features import features, projection_features, soma_features
from .utilities import *
from .ml_utilities import *

neurite_types = ['(basal) dendrite', 'apical dendrite', 'axon', 'soma']

package_path = os.path.realpath(__file__).replace("__init__.py", "")
# Temporary solution
annotation = image(package_path+"data/annotation_10.nrrd")
bs = brain_structure(package_path+"data/Mouse.csv")
bs.get_selected_regions(package_path+"data/CCFv3 Summary Structures.xlsx")
