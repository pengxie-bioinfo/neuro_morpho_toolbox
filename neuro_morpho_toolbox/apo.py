import numpy as np
import pandas as pd

# from .utilities import cart2pol_3d
# from .brain_structure import *
from neuro_morpho_toolbox import annotation
from neuro_morpho_toolbox import bs

# For testing purpose
import time

# Global variables
type_dict = {"soma": 1,
             "axon": 2,
             "(basal) dendrite": 3,
             "apical dendrite": 4
             }
# midline = annotation.size['z'] / 2

marker_header = ["##x", "y", "z", "radius", "shape", "name", "comment", "color_r", "color_g", "color_b"]
apo_header = ["##n", "orderinfo", "name", "comment", "z", "x", "y", "pixmax", "intensity", "sdev", "volsize", "mass",
              "", "", "", "color_r", "color_g", "color_b"]
common_header = ["n", "name", "x", "y", "z", "comment", "color_r", "color_g", "color_b",
                 "radius", "shape",
                 "orderinfo", "pixmax", "intensity", "sdev", "volsize", "mass"
                 ]
default_dict = {'radius': 1,
                'shape': 1,
                'pixmax': 0,
                'intensity': 0,
                'sdev': 0,
                'volsize': 50,
                'mass': 0
                }


def read_apo(input_file):
    '''
    Read .apo file
    :param input_file:
    :return:
    '''
    n_skip = 0
    with open(input_file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("#"):
                n_skip += 1
            else:
                break
    apo_read = pd.read_csv(input_file, skiprows=n_skip, sep=",",
                           names=["n", "orderinfo", "name", "comment", "z", "x", "y", "pixmax", "intensity", "sdev",
                                  "volsize", "mass", "", "", "", "color_r", "color_g", "color_b"]
                           )
    res = pd.DataFrame(index=apo_read.index,
                       columns=common_header
                       )
    for cur_col in res.columns.tolist():
        if cur_col in apo_read.columns.tolist():
            res[cur_col] = apo_read[cur_col]
        else:
            res[cur_col] = default_dict[cur_col]
    return res

def read_marker(input_file):
    '''
    Read .marker file
    :param input_file:
    :return:
    '''


class marker:
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = pd.DataFrame()
        if input_file.lower().endswith(".apo"):
            self.df = read_apo(input_file)
        elif input_file.lower().endswith(".marker"):
            self.df = read_marker(input())
        else:
            print("Error: invalid suffix of input file. Required suffix: '.apo' or '.marker'.")
            return
        return

    def save_apo(self, output_file):
        tp = self.df.copy()
        tp.rename(columns={'n': '##n'}, inplace=True)
        res = pd.DataFrame(index=self.df.index,
                           columns=apo_header
                           )
        for cur_col in res.columns.tolist():
            if cur_col in tp.columns.tolist():
                res[cur_col] = tp[cur_col]
            elif cur_col in list(default_dict.keys()):
                res[cur_col] = default_dict[cur_col]
        res.to_csv(output_file, index=False, sep=',')
        return

    def scale(self, xyz_scales, inplace=True):
        if inplace:
            self.df["x"] = self.df["x"] * xyz_scales[0]
            self.df["y"] = self.df["y"] * xyz_scales[1]
            self.df["z"] = self.df["z"] * xyz_scales[2]
            return self.df
        else:
            res = self.df.copy()
            res["x"] = self.df["x"] * xyz_scales[0]
            res["y"] = self.df["y"] * xyz_scales[1]
            res["z"] = self.df["z"] * xyz_scales[2]
            return res

    def shift(self, xyz_shift, inplace=True):
        if inplace:
            self.df["x"] = self.df["x"] + xyz_shift[0]
            self.df["y"] = self.df["y"] + xyz_shift[1]
            self.df["z"] = self.df["z"] + xyz_shift[2]
            return self.df
        else:
            res = self.df.copy()
            res["x"] = res["x"] + xyz_shift[0]
            res["y"] = res["y"] + xyz_shift[1]
            res["z"] = res["z"] + xyz_shift[2]
            return res

    def get_regions(self):
        self.df['Region'] = "unknown"
        self.df['Region_id'] = 0
        tp = pd.DataFrame({"x": np.array(np.array(self.df.x) / annotation.space['x'], dtype="int32"),
                           "y": np.array(np.array(self.df.y) / annotation.space['y'], dtype="int32"),
                           "z": np.array(np.array(self.df.z) / annotation.space['z'], dtype="int32"),
                           }, index=self.df.index)
        tp = tp[((tp.x >= 0) & (tp.x < annotation.size['x']) &
                 (tp.y >= 0) & (tp.y < annotation.size['y']) &
                 (tp.z >= 0) & (tp.z < annotation.size['z'])
                 )]
        for cname in tp.index.tolist():
            cur_id = annotation.array[tp.loc[cname, 'x'],
                                      tp.loc[cname, 'y'],
                                      tp.loc[cname, 'z']
            ]
            if cur_id in list(bs.dict_to_selected.keys()):
                cur_id = bs.dict_to_selected[cur_id]
                self.df.loc[cname, 'Region'] = bs.id_to_name(cur_id)
                self.df.loc[cname, 'Region_id'] = cur_id
        return self.df