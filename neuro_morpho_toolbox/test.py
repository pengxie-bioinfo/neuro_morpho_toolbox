import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import neuro_morpho_toolbox as nmt

# swc_file = "./neuro_morpho_toolbox/data/236174_03029_03128_X13178_Y25409_QCed.swc"
swc_file = "./neuro_morpho_toolbox/data/236174_03429_03528_X12632_Y10625_QCed.swc"
neuron = nmt.swc(swc_file)
segment = neuron.get_segments()

annotation = nmt.image("./neuro_morpho_toolbox/data/annotation_10.nrrd")
bs = nmt.brain_structure("./neuro_morpho_toolbox/data/Mouse.csv")
bs.get_selected_regions("./neuro_morpho_toolbox/data/CCFv3 Summary Structures.xlsx")


region_df = neuron.get_region_matrix(annotation, bs)
region_df.index = ["_".join([str(region_df.loc[i, "structure_id"]), str(region_df.loc[i, "hemisphere_id"])]) for i in region_df.index.tolist()]


brain_levels = pd.read_csv("./neuro_morpho_toolbox/data/Mouse.csv", index_col=[0], usecols=[1,2])
brain_levels.columns = ["Abbrevation"]
region_df["Abbrevation"] = brain_levels.loc[region_df.structure_id, "Abbrevation"].tolist()
region_df = region_df.sort_values("axon", ascending=False)



ref_region_df = pd.read_csv(swc_file+".csv", index_col=0)
lab = [i for i,j in enumerate(ref_region_df.structure_id.tolist()) if j in bs.selected_regions]
ref_region_df = ref_region_df.iloc[lab, :]
ref_region_df.index = ["_".join([str(ref_region_df.loc[i, "structure_id"]), str(ref_region_df.loc[i, "hemisphere_id"])]) for i in ref_region_df.index.tolist()]
ref_region_df["Abbrevation"] = brain_levels.loc[ref_region_df.structure_id, "Abbrevation"].tolist()
ref_region_df = ref_region_df.sort_values("axon", ascending=False)



common_region = list(set(region_df.index.tolist()).intersection(set(ref_region_df.index.tolist())))
diff_region = list(set(region_df.index.tolist()).difference(set(ref_region_df.index.tolist())))
plt.scatter(region_df.loc[common_region, "axon"], ref_region_df.loc[common_region, "axon"])

region_df[["axon", "(basal) dendrite", "Abbrevation"]]