import numpy as np
import pandas as pd

from .utilities import cart2pol_3d
from .brain_structure import *
from neuro_morpho_toolbox import annotation
from neuro_morpho_toolbox import bs

# For testing purpose
import time

# Global variables
type_dict = {"soma":1,
             "axon":2,
             "(basal) dendrite":3,
             "apical dendrite":4
            }
midline = annotation.size['z'] / 2


class neuron:
    def __init__(self, file, zyx=False):
        self.file = file
        self.name = file.split("/")[-1].split(".")[0]

        n_skip = 0
        with open(self.file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith("#"):
                    n_skip += 1
                else:
                    break
        f.close()
        if zyx:
            names = ["##n", "type", "z", "y", "x", "r", "parent"]
        else:
            names = ["##n", "type", "x", "y", "z", "r", "parent"]
        swc = pd.read_csv(self.file, index_col=0, skiprows=n_skip, sep=" ",
                          usecols=[0, 1, 2, 3, 4, 5, 6],
                          names=names
                          )
        self.swc = swc
        _ = self.get_soma()
        return

    def get_soma(self):
        soma = self.swc[((self.swc.type==1) & (self.swc.parent==-1))]
        if len(soma)!=1:
            print(("Invalid number of soma found: %d" % len(soma)))
            self.soma = pd.DataFrame({"x": np.nan,
                                      "y": np.nan,
                                      "z": np.nan}, index=[self.name])
        else:
            soma = pd.DataFrame({"x": soma.x.iloc[0],
                                 "y": soma.y.iloc[0],
                                 "z": soma.z.iloc[0]}, index=[self.name])
            self.soma = soma
            if soma.loc[self.name, "z"] < (annotation.micron_size["z"]/2):
                self.hemi = 1
            else:
                self.hemi = 2
        return self.soma

    def pass_qc(self):
        '''
        :return: True or False
        '''
        # unique soma
        if self.soma.isnull().values.any():
            return False
        if not ((self.soma.loc[self.name, "x"] >= 0) & (self.soma.loc[self.name, "x"] < annotation.micron_size['x']) &
                 (self.soma.loc[self.name, "y"] >= 0) & (self.soma.loc[self.name, "y"] < annotation.micron_size['y']) &
                 (self.soma.loc[self.name, "z"] >= 0) & (self.soma.loc[self.name, "z"] < annotation.micron_size['z'])
                ):
            return False
        return True


    def get_segments(self):
        # lab = [i for i,name in enumerate(self.swc.index.tolist()) if self.swc.loc[name, "parent"]!=(-1)]
        child = self.swc[self.swc.parent != (-1)]
        parent = self.swc.loc[child.parent]
        rho, theta, phi = cart2pol_3d(np.array(child[["x", "y", "z"]]) - np.array(parent[["x", "y", "z"]]))
        res = pd.DataFrame({"type": child.type,
                            "rho": rho,
                            "theta": theta,
                            "phi": phi,
                            "x": (np.array(child.x) + np.array(parent.x)) / 2,
                            "y": (np.array(child.y) + np.array(parent.y)) / 2,
                            "z": (np.array(child.z) + np.array(parent.z)) / 2
                            })
        res.index = range(1, len(child)+1)
        # soma
        soma = self.swc[((self.swc.type==1) & (self.swc.parent==-1))]
        if len(soma)>0:
            soma_res = pd.DataFrame({"type": 1,
                                     "rho": 1,
                                     "theta": 0,
                                     "phi": 0,
                                     "x": soma.x.iloc[0],
                                     "y": soma.y.iloc[0],
                                     "z": soma.z.iloc[0]},
                                    index=[0])
            res = soma_res.append(res)
        return res
    def get_region_matrix(self, annotation, brain_structure, region_used=None):
        start = time.time()
        segment = self.get_segments()

        tp = pd.DataFrame({"x": np.array(np.array(segment.x) / annotation.space['x'], dtype="int32"),
                           "y": np.array(np.array(segment.y) / annotation.space['y'], dtype="int32"),
                           "z": np.array(np.array(segment.z) / annotation.space['z'], dtype="int32"),
                           "rho": segment.rho,
                           "type": segment.type
                           })
        tp = tp[((tp.x >= 0) & (tp.x < annotation.size['x']) &
                 (tp.y >= 0) & (tp.y < annotation.size['y']) &
                 (tp.z >= 0) & (tp.z < annotation.size['z'])
                )]

        # Add region id to the dataframe
        if region_used is None:
            region_used = bs.selected_regions
            dict_to_used = bs.dict_to_selected
        else:
            assert all([(i in brain_structure.level.index.tolist()) for i in region_used]), "Given regions invalid. Please check 'region_used'."
            dict_to_used = {}
            for cur_region in region_used:
                child_ids = bs.get_all_child_id(cur_region)
                for i in child_ids:
                    dict_to_used[i] = cur_region

        tp["region_id"] = annotation.array[tp.x, tp.y, tp.z]
        tp = tp[tp.region_id.isin(list(dict_to_used.keys()))]
        tp["region_id"] = [dict_to_used[i] for i in tp["region_id"].tolist()]

        # Get output dataframe
        res = pd.DataFrame(columns=["structure_id",
                                    "hemisphere_id",
                                    "soma",
                                    "axon",
                                    "(basal) dendrite",
                                    "apical dendrite"
                                    ])

        hemi_1 = tp[tp.z < midline]
        hemi_2 = tp[tp.z >= midline]
        for cur_hemi_id, cur_hemi in zip([1,2], [hemi_1, hemi_2]):
            cur_df = pd.DataFrame({"structure_id": region_used,
                                   "hemisphere_id": [cur_hemi_id] * len(region_used),
                                   "soma": [0] * len(region_used),
                                   "axon": [0] * len(region_used),
                                   "(basal) dendrite": [0] * len(region_used),
                                   "apical dendrite": [0] * len(region_used)
                                   },
                                  index=region_used
                                  )
            for type_name, i in type_dict.items():
                cur_hemi_type = cur_hemi[cur_hemi.type == i].groupby(['region_id'])['rho'].sum()
                cur_df.loc[cur_hemi_type.index, type_name] = cur_hemi_type
            res = pd.concat([res, cur_df], axis=0)
        # print("Time elapsed by get_region_matrix: %.2f; %s" % (time.time() - start, self.name))
        # res = res[np.sum(res[["axon", "apical dendrite", "(basal) dendrite"]], axis=1)>0]
        return res
    # def get_region_matrix(self, annotation, brain_structure, region_used=None):
    #     segment = self.get_segments()
    #
    #     tp = pd.DataFrame({"x": np.array(np.array(segment.x) / annotation.space['x'], dtype="int32"),
    #                        "y": np.array(np.array(segment.y) / annotation.space['y'], dtype="int32"),
    #                        "z": np.array(np.array(segment.z) / annotation.space['z'], dtype="int32"),
    #                        "rho": segment.rho,
    #                        "type": segment.type
    #                        })
    #     tp = tp[((tp.x >= 0) & (tp.x < annotation.size['x']) &
    #              (tp.y >= 0) & (tp.y < annotation.size['y']) &
    #              (tp.z >= 0) & (tp.z < annotation.size['z'])
    #             )]
    #     # print(np.max(tp[["x", "y", "z"]]))
    #     # assert (all(tp.x >= 0) & all(tp.x < annotation.size[0]) &
    #     #         all(tp.y >= 0) & all(tp.y < annotation.size[1]) &
    #     #         all(tp.z >= 0) & all(tp.z < annotation.size[2])
    #     #         ), "Error: SWC segments out of range."
    #
    #
    #
    #     if region_used is None:
    #         region_used = bs.selected_regions
    #         dict_to_used = bs.dict_to_selected
    #     else:
    #         assert all([(i in brain_structure.level.index.tolist()) for i in region_used]), "Given regions invalid. Please check 'region_used'."
    #         dict_to_used = {}
    #         for cur_region in region_used:
    #             child_ids = bs.get_all_child_id(cur_region)
    #             for i in child_ids:
    #                 dict_to_used[i] = cur_region
    #
    #     tp["region_id"] = annotation.array[tp.x, tp.y, tp.z]
    #     tp = tp[tp.region_id.isin(list(dict_to_used.keys()))]
    #     tp["region_id"] = [dict_to_used[i] for i in annotation.array[tp.x, tp.y, tp.z].tolist()]
    #
    #     midline = annotation.size['z']/ 2
    #     # Get output dataframe
    #     res = pd.DataFrame({"structure_id": np.append(region_used, region_used),
    #                         "hemisphere_id": [1]*len(region_used) + [2]*len(region_used)
    #                         })
    #     soma = []
    #     axon = []
    #     basal_dendrite = []
    #     apical_dendrite = []
    #
    #     hemi_1 = tp[tp.z < midline]
    #     hemi_2 = tp[tp.z >=midline]
    #     start = time.time()
    #     for cur_hemi in [hemi_1, hemi_2]:
    #         for ct, cur_region in enumerate(region_used):
    #             child_ids = brain_structure.get_all_child_id(cur_region)
    #             # print("%d/%d\t%s\t%d" % (ct + 1,
    #             #                          len(region_used),
    #             #                          brain_structure.level.loc[cur_region, "Abbrevation"],
    #             #                          len(child_ids)))
    #             idx = []
    #             for i in child_ids:
    #                 idx = idx + cur_hemi[cur_hemi.region_id == i].index.tolist()
    #             temp = cur_hemi.loc[idx]
    #             soma.append(np.sum(temp[temp.type == 1]["rho"]))
    #             axon.append(np.sum(temp[temp.type == 2]["rho"]))
    #             basal_dendrite.append(np.sum(temp[temp.type == 3]["rho"]))
    #             apical_dendrite.append(np.sum(temp[temp.type == 4]["rho"]))
    #     res["soma"] = soma
    #     res["axon"] = axon
    #     res["(basal) dendrite"] = basal_dendrite
    #     res["apical dendrite"] = apical_dendrite
    #     print("Time elapsed by region_count: %.2f" % (time.time() - start))
    #     # res = res[np.sum(res[["axon", "apical dendrite", "(basal) dendrite"]], axis=1)>0]
    #     return res

    def get_region_sum(self, annotation, brain_structure, region_name):
        structure_id = brain_structure.name_to_id(region_name)
        if structure_id == -1:
            return pd.DataFrame()
        return self.get_region_matrix(annotation, brain_structure, [structure_id])

    def flip(self, axis, axis_max):
        self.swc[axis] = axis_max - self.swc[axis]
        return

    def scale(self, xyz_scales, inplace=True):
        if inplace:
            self.swc["x"] = self.swc["x"] * xyz_scales[0]
            self.swc["y"] = self.swc["y"] * xyz_scales[1]
            self.swc["z"] = self.swc["z"] * xyz_scales[2]
            return self.swc
        else:
            res = self.swc.copy()
            res["x"] = self.swc["x"] * xyz_scales[0]
            res["y"] = self.swc["y"] * xyz_scales[1]
            res["z"] = self.swc["z"] * xyz_scales[2]
            return res

    def shift(self, xyz_shift, inplace=True):
        if inplace:
            self.swc["x"] = self.swc["x"] + xyz_shift[0]
            self.swc["y"] = self.swc["y"] + xyz_shift[1]
            self.swc["z"] = self.swc["z"] + xyz_shift[2]
            return self.swc
        else:
            res = self.swc.copy()
            res["x"] = res["x"] + xyz_shift[0]
            res["y"] = res["y"] + xyz_shift[1]
            res["z"] = res["z"] + xyz_shift[2]
            return res

    def save(self, file_name):
        self.swc.to_csv(file_name, sep=" ")
        return

