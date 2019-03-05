import numpy as np
import pandas as pd

from .utilities import cart2pol_3d

class swc:
    def __init__(self, file):
        self.file = file
        self.name = file.split(".")[0]

        n_skip = 0
        with open(self.file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith("#"):
                    n_skip += 1
                else:
                    break
        f.close()
        swc = pd.read_csv(self.file, index_col=0, skiprows=n_skip, sep=" ",
                          usecols=[0, 1, 2, 3, 4, 5, 6],
                          names=["", "type", "x", "y", "z", "r", "parent"]
                          )
        self.swc = swc
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
        return res
    def get_region_matrix(self, annotation, brain_structure):
        segment = self.get_segments()
        tp = pd.DataFrame({"x": np.array(np.around(np.array(segment.x) / annotation.space[0]), dtype="int32"),
                           "y": np.array(np.around(np.array(segment.y) / annotation.space[1]), dtype="int32"),
                           "z": np.array(np.around(np.array(segment.z) / annotation.space[2]), dtype="int32"),
                           "rho": segment.rho,
                           "type": segment.type
                           })
        # print(np.max(tp[["x", "y", "z"]]))
        assert (all(tp.x >= 0) & all(tp.x < annotation.size[0]) &
                all(tp.y >= 0) & all(tp.y < annotation.size[1]) &
                all(tp.z >= 0) & all(tp.z < annotation.size[2])
                ), "Error: SWC segments out of range."
        tp["region_id"] = annotation.array[tp.z, tp.y, tp.x]
        # region_used = list(set(tp.region_id.tolist()))
        region_used = brain_structure.selected_regions

        midline = annotation.size[2] / 2
        # Get output dataframe
        res = pd.DataFrame({"structure_id": np.append(region_used, region_used),
                            "hemisphere_id": [1]*len(region_used) + [2]*len(region_used)
                            })
        axon = []
        basal_dendrite = []
        apical_dendrite = []
        # Hemisphere 1
        hemi_1 = tp[tp.z < midline]
        hemi_2 = tp[tp.z >=midline]
        for ct, cur_region in enumerate(brain_structure.selected_regions):
            child_ids = brain_structure.get_all_child_id(cur_region)
            print("%d/%d\t%s\t%d" % (ct+1,
                                     len(brain_structure.selected_regions),
                                     brain_structure.level.loc[cur_region, "Abbrevation"],
                                     len(child_ids)))
            idx = []
            for i in child_ids:
                idx = idx + hemi_1[hemi_1.region_id == i].index.tolist()
            temp = hemi_1.loc[idx]
            axon.append(np.sum(temp[temp.type == 2]["rho"]))
            basal_dendrite.append(np.sum(temp[temp.type == 3]["rho"]))
            apical_dendrite.append(np.sum(temp[temp.type == 4]["rho"]))
        # Hemisphere 2
        for ct, cur_region in enumerate(brain_structure.selected_regions):
            child_ids = brain_structure.get_all_child_id(cur_region)
            print("%d/%d\t%s\t%d" % (ct+1,
                                     len(brain_structure.selected_regions),
                                     brain_structure.level.loc[cur_region, "Abbrevation"],
                                     len(child_ids)))
            idx = []
            for i in child_ids:
                idx = idx + hemi_2[hemi_2.region_id == i].index.tolist()
            temp = hemi_2.loc[idx]
            axon.append(np.sum(temp[temp.type == 2]["rho"]))
            basal_dendrite.append(np.sum(temp[temp.type == 3]["rho"]))
            apical_dendrite.append(np.sum(temp[temp.type == 4]["rho"]))
        res["axon"] = axon
        res["(basal) dendrite"] = basal_dendrite
        res["apical dendrite"] = apical_dendrite
        res = res[np.sum(res[["axon", "apical dendrite", "(basal) dendrite"]], axis=1)>0]

        return res