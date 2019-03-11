import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Image process
import SimpleITK as sitk

from .swc import swc
from .image import image
from .brain_structure import brain_structure
from .neuron_features import features, projection_features
from .utilities import *
from .ml_utilities import *

neurite_types = ['(basal) dendrite', 'apical dendrite', 'axon', 'soma']

# Temporary solution
annotation = image("/local1/Documents/python/neuro_morpho_toolbox/neuro_morpho_toolbox/data/annotation_10.nrrd")
bs = brain_structure("/local1/Documents/python/neuro_morpho_toolbox/neuro_morpho_toolbox/data/Mouse.csv")
bs.get_selected_regions("/local1/Documents/python/neuro_morpho_toolbox/neuro_morpho_toolbox/data/CCFv3 Summary Structures.xlsx")
