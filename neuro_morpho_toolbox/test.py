import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import pickle

import neuro_morpho_toolbox as nmt


# 1. read swc
swc_file = "/local1/Documents/python/neuro_morpho_toolbox/neuro_morpho_toolbox/data/Janelia_1000/All/AA0001.swc"
neuron = nmt.swc(swc_file, zyx=False)
neuron.flip("z", axis_max=nmt.annotation.micron_size['z'])
neuron.scale([1/10, 1/10, 1/10])
neuron.save("/local1/Documents/python/neuro_morpho_toolbox/neuro_morpho_toolbox/data/temp_swc/AA0001.swc")

swc_file = "/local1/Documents/CLA/data/CCF/mapped/swc/Whole/236174_03229_03312_X11951_Y11316_QCed.swc"
neuron = nmt.swc(swc_file, zyx=False)
neuron.scale([1/10, 1/10, 1/10])
neuron.save("/local1/Documents/python/neuro_morpho_toolbox/neuro_morpho_toolbox/data/temp_swc/236174_03229_03312_X11951_Y11316_QCed.swc")

#


########################################################################
# Validation 1: comparing with Allen pipeline
########################################################################

swc_path = "/local1/Documents/CLA/data/CCF/mapped/swc/Whole/"
swc_files = nmt.get_sample_list(swc_path, "swc")
df_list = []
ref_df_list = []
for swc_file in sorted(swc_files)[:5]:
    print(len(df_list)+1, swc_file)

    # 1.1 results of current pipeline
    swc_file = swc_path + swc_file
    csv_file = swc_file.replace("swc/Whole", "location") + ".csv"
    neuron = nmt.swc(swc_file)
    # neuron.flip("y", axis_max=nmt.annotation.size['y']*nmt.annotation.space['y'])

    region_df = neuron.get_region_matrix(nmt.annotation, nmt.bs)
    region_df = region_df[np.sum(region_df[["axon", "apical dendrite", "(basal) dendrite"]], axis=1) > 0]
    region_df.index = ["_".join([str(region_df.loc[i, "structure_id"]), str(region_df.loc[i, "hemisphere_id"])]) for i
                       in region_df.index.tolist()]
    region_df["Abbrevation"] = nmt.bs.level.loc[region_df.structure_id, "Abbrevation"].tolist()
    region_df = region_df.sort_values("axon", ascending=False)

    # 1.2 results from Allen pipeline
    ref_region_df = pd.read_csv(csv_file, index_col=0)
    lab = [i for i, j in enumerate(ref_region_df.structure_id.tolist()) if j in nmt.bs.selected_regions]
    ref_region_df = ref_region_df.iloc[lab, :]
    ref_region_df.index = [
        "_".join([str(ref_region_df.loc[i, "structure_id"]), str(ref_region_df.loc[i, "hemisphere_id"])]) for i in
        ref_region_df.index.tolist()]
    ref_region_df["Abbrevation"] = nmt.bs.level.loc[ref_region_df.structure_id, "Abbrevation"].tolist()
    ref_region_df = ref_region_df.sort_values("axon", ascending=False)
    df_list.append(region_df)
    ref_df_list.append(ref_region_df)

# Linear regression

ct_list = []
regression_list = []
max_x = 0
for region_df, ref_region_df in zip(df_list, ref_df_list):
    common_region = list(set(region_df.index.tolist()).intersection(set(ref_region_df.index.tolist())))
    extra_region = list(set(region_df.index.tolist()).difference(set(ref_region_df.index.tolist())))
    missing_region = list(set(ref_region_df.index.tolist()).difference(set(region_df.index.tolist())))
    ct_list = ct_list + [len(common_region), len(extra_region), len(missing_region)]

    X = np.array(region_df.fillna(0).loc[common_region, "axon"]).reshape(-1, 1)
    y = np.array(ref_region_df.fillna(0).loc[common_region, "axon"]).reshape(-1, 1)
    plt.scatter(X, y)
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    regression_list = regression_list + [reg.score(X, y), reg.coef_[0, 0]]
    if max_x < np.max(X):
        max_x = np.max(X)

regression_df = pd.DataFrame(np.array(regression_list).reshape(-1,2), columns=["Score", "Coef"])
plt.plot([0, max_x], [0, max_x * np.mean(regression_df.Coef)], linestyle="--", color='black')

plt.xlabel("Axon length (um) new")
plt.ylabel("Axon volume (not sure) old")

plt.text(np.dot(np.array(plt.xlim()), np.array([0.90,0.10])),
         np.dot(np.array(plt.ylim()), np.array([0.10,0.90])),
         "Slope: %0.2f" % np.mean(regression_df.Coef) + r'$\pm$' + "%0.2f" % np.std(regression_df.Coef)
         )
plt.text(np.dot(np.array(plt.xlim()), np.array([0.90,0.10])),
         np.dot(np.array(plt.ylim()), np.array([0.20,0.80])),
         "Score: %0.5f" % np.mean(regression_df.Score)
         )

#########################################################################
# Validation 2: comparing with Janelia 1000 reconstruction
#########################################################################
swc_path = "/local1/Documents/python/neuro_morpho_toolbox/neuro_morpho_toolbox/data/Janelia_1000/L5PT/"
# swc_path = "/local1/Documents/python/neuro_morpho_toolbox/neuro_morpho_toolbox/data/Janelia_1000/L6CT/"
swc_files = nmt.get_sample_list(swc_path, "swc")

TH_sum = []
for i, swc_file in enumerate(swc_files):
    print(i, swc_file)
    swc_file = swc_path + swc_file
    neuron = nmt.swc(swc_file, zyx=False)
    neuron.flip("z", axis_max=nmt.annotation.size['z']*nmt.annotation.space['z'])
    tp = neuron.get_region_sum(nmt.annotation, nmt.bs, "Thalamus")
    TH_sum.append(np.sum(tp.axon))

print("%0.2f + %0.2f" % (np.mean(TH_sum)/1000, np.std(TH_sum)/1000))