
.. code:: ipython3

    import neuro_morpho_toolbox as nmt
    
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    import os
    import re
    import pickle
    # from timeit import default_timer as timer
    # from sklearn.preprocessing import scale
    # from sklearn.manifold import Isomap, TSNE
    # from sklearn.cluster import KMeans
    # from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    # import seaborn as sns
    # from scipy.stats import mannwhitneyu
    
    # # from pysankey import sankey
    
    # from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering, Birch
    # from sklearn.neighbors import NearestNeighbors
    # from sklearn.feature_selection import mutual_info_classif
    # from sklearn import metrics
    # import igraph
    # from math import ceil
    # from timeit import default_timer as timer
    # import subprocess
    # from pathlib import Path
    # from numpy import linalg as LA
    
    # plot nrrd
    import SimpleITK as sitk
    
    %matplotlib inline


.. code:: ipython3

    def get_location_tables_by_path(swc_path, loc_path, soma_path, vis_path, zyx, flip_axes=[]):
        swc_files = nmt.get_sample_list(swc_path, "swc")
        for i,swc_file in enumerate(swc_files):
            print("%d/%d %s" % (i+1, len(swc_files), swc_file))
            neuron = nmt.swc(swc_path + swc_file, zyx=zyx)
            for flip_axis in flip_axes:
                neuron.flip(flip_axis, 
                            axis_max=nmt.annotation.size[flip_axis]*nmt.annotation.space[flip_axis]
                           )
            # 1. Soma (XYZ) df
            soma_df = neuron.get_segments()
            if 0 in soma_df.index.tolist():
                if soma_df.loc[0, "type"]==1:
                    soma_df = pd.DataFrame(soma_df.loc[0, ["x", "y", "z"]]).transpose()
                    soma_df.to_csv(soma_path+swc_file+".csv")
            else:
                print("Soma not found:\t%s" % swc_file)
            
            # 2. location df (including axon, soma, dendrite)
            region_df = neuron.get_region_matrix(nmt.annotation, nmt.bs)
            # Select non-zero regions to export
            region_df = region_df[np.sum(region_df[["soma", "axon", "apical dendrite", "(basal) dendrite"]], axis=1) > 0]
            # Add region names for reading
            region_df["Abbrevation"] = nmt.bs.level.loc[region_df.structure_id, "Abbrevation"].tolist()
            # Sort for reading
            region_df = region_df.sort_values(["soma", "(basal) dendrite", "axon"], ascending=False)
            region_df.index = range(len(region_df))
            region_df.to_csv(loc_path+swc_file+".csv")
            # Save for visualization
            neuron.scale([1/10, 1/10, 1/10])
            neuron.save(vis_path + swc_file)
        return


fMOST\_Claustrum
----------------

.. code:: ipython3

    swc_path = "../data/CCF/mapped/swc/Whole/"
    loc_path = "../data/CCF/mapped/location/Whole/"
    soma_path = "../data/CCF/mapped/location/Soma/"
    vis_path = "../data/CCF/mapped/swc/Visualization/"
    zyx=False
    flip_axes = []
    
    get_location_tables_by_path(swc_path, loc_path, soma_path, vis_path, zyx, flip_axes)


.. parsed-literal::

    1/100 236174_04229_04328_X13663_Y8589_QCed.swc
    2/100 17109_2601-2700-X10213-Y8783_QCed.ano.eswc
    3/100 17782_3651_X35286_Y18512_QCed.swc
    4/100 236174_4429_04528_X13147_Y8003_QCed.swc
    5/100 17781_6228_x12697_y8412_QCed.swc
    6/100 236174_3729_03828_X15151_Y26698_QCed.swc
    7/100 17109_6301_6400_X4756_Y24516_QCed.swc
    8/100 236174_03229_03328_X11950_Y11335_QCed.swc
    9/100 236174_3970_04170_X13439_Y8678_QCed.swc
    10/100 236174_3729_03828_X13645_Y9551_QCed.swc
    11/100 17109_3701_03800_X9228_Y26684_QCed.swc
    12/100 17109_7001-7100-X5738-Y6470_QCed.ano.eswc
    13/100 17781_3139_X20033_Y17506_QCed.swc
    14/100 17109_2301_2400_X8535_Y23051_QCed.swc
    15/100 17781_3668_x9453_y17266_QCed.swc
    16/100 236174_7077_07089_X14579_Y30892_QCed.swc
    17/100 236174_03929_04028_X12721_Y8845_QCed.swc
    18/100 17109_01901_02000_X9602_Y10508_QCed.swc
    19/100 17109_1801_1900_X6698_Y12550_QCed.swc
    20/100 17109_2601_2700_X9498_Y8169_QCed.swc
    21/100 236174_3829_03928_X16301_Y26647_QCed.swc
    22/100 236174_03529_03628_X13394_Y26567_QCed.swc
    23/100 17781_6866_x8954_y41812_QCed.swc
    24/100 236174_5138_05237_X16501_Y28259_QCed.swc
    25/100 17109_7001_7100_X6205_Y5194_QCed.swc
    26/100 236174_03447_03459_X12562_Y10626_QCed.swc
    27/100 17109_6401_6500_X7989_Y3767_QCed.swc
    28/100 236174_03729_03828_X12692_Y9419_QCed.swc
    29/100 236174_4029_04128_X13079_Y8858_QCed.swc
    30/100 17781_6028_X9772_Y42888_QCed.swc
    31/100 236174_03229_03328_X12413_Y11831_QCed.swc
    32/100 17781_6202_x7633_y12296_QCed.swc
    33/100 17109_6801_06900_X7432_Y4405_QCed.swc
    34/100 17109_6201_6300_X4328_Y6753_QCed.swc
    35/100 17109_6401_6500_X7641_Y3978_QCed.swc
    36/100 17782_3487_X11014_Y17041_QCed.swc
    37/100 236174_6855_06875_X15550_Y29832_QCed.swc
    38/100 236174_3729_03828_X15443_Y26410_QCed.swc
    39/100 236174_02657_02671_X11930_Y12250_QCed.swc
    40/100 236174_5670_05870_X14073_Y29439_QCed.swc
    41/100 17109_2401_2500_X9954_Y9122_QCed.swc
    42/100 17109_6601_6700_X4384_Y7436_QCed.swc
    43/100 236174_03529_03628_X13688_Y26111_QCed.swc
    44/100 17109_01701_01800_X8048_Y22277_QCed.swc
    45/100 17109_2401_2500_X8977_24184_QCed.swc
    46/100 236174_05338_05437_X13590_Y7348_QCed.swc
    47/100 236174_3893_3908_X17507_Y26071_QCed.swc
    48/100 236174_5840_06040_X15240_Y29741_QCed.swc
    49/100 236174_4729_04829_X16869_Y27809_QCed.swc
    50/100 236174_3829_03928_X13590_Y9284_QCed.swc
    51/100 236174_05738_05837_X11712_Y6818_QCed.swc
    52/100 17782_3253_X33739_Y18314_QCed.swc
    53/100 17781_6287_x8230_y11645_QCed.swc
    54/100 236174_6338_06437_X12092_Y5845_QCed.swc
    55/100 17782_3138_X32805_Y18352_QCed.swc
    56/100 236174_04266_X13848_Y8550_Finalized_QCed.swc
    57/100 17109_6601_6700_X5417_Y25287_QCed.swc
    58/100 17109_4101_4200_X6753_Y6197_QCed.swc
    59/100 17781_3320_X16423_Y14748_QCed.swc
    60/100 236174_03329_03428_X13938_Y26099_QCed.swc
    61/100 236174_03536_03545_X15159_Y25525_QCed.swc
    62/100 17781_6627_x22762_y14486_QCed.swc
    63/100 17781_6643_x22317_y12822_QCed.swc
    64/100 17109_6901_7000_X7203_Y26714_QCed.swc
    65/100 17782_3284_X11909_Y16428_QCed.swc
    66/100 236174_3215_X11999_Y11133_QCed.swc
    67/100 236174_6338_06437_X12496_Y6617_QCed.swc
    68/100 236174_04129_04228_X16214_Y10304_QCed.swc
    69/100 236174_03229_03428_X11884_Y10380_QCed.swc
    70/100 17109_6501_6600_X6997_Y4287_QCed.swc
    71/100 17781_5166_x11993_y10858_QCed.swc
    72/100 17781_3668_X17825_Y13313_QCed.swc
    73/100 17109_2401_2500_X9338_Y2394_QCed.swc
    74/100 17781_4698_X17857_Y11456_QCed.swc
    75/100 236174_03429_03528_X12632_Y10625_QCed.swc
    76/100 236174_02669_02678_X10674_Y13239_QCed.swc
    77/100 236174_03929_04028_X13599_Y9165_QCed.swc
    78/100 17781_4095_x17570_y12460_QCed.swc
    79/100 236174_03029_03128_X12820_Y24699_QCed.swc
    80/100 17781_6626_x22318_y13682_QCed.swc
    81/100 17782_3672_X34784_Y19432_QCed.swc
    82/100 17781_6881_x22248_y12698_QCed.swc
    83/100 236174_3829_03928_X14826_Y27255_QCed.swc
    84/100 17109_2401_2500_X9695_Y9693_QCed.swc
    85/100 236174_4647_X16405_Y27845_QCed.swc
    86/100 236174_03229_03312_X11951_Y11316_QCed.swc
    87/100 236174_03029_03128_X13178_Y25409_QCed.swc
    88/100 236174_03001_03008_X12887_Y24248_QCed.swc
    89/100 17781_4874_x11466_y11924_QCed.swc
    90/100 17109_3101_3200_X10824_Y7188_QCed.swc
    91/100 236174_03629_03728_X15224_Y26052_QCed.swc
    92/100 17781_5894_x11460_y10079_QCed.swc
    93/100 17781_2881_X4240_Y36304_QCed.swc
    94/100 17109_2301_2400_X9418_23665_QCed.swc
    95/100 236174_6438_06537_X12218_Y5897_QCed.swc
    96/100 17781_6270_x9494_y43380_QCed.swc
    97/100 236174_03529_03628_X12805_Y10541_QCed.swc
    98/100 17782_3352_X11384_Y16404_QCed.swc
    99/100 17781_5655_x10641_y11102_QCed.swc
    100/100 17781_6325_x14423_y8278_QCed.swc


.. code:: ipython3

    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    plot_region = ["ACAd5", "RSPv5"]
    for i, cur_region in enumerate(plot_region):
        structure_id = nmt.bs.name_to_id(cur_region)
        if structure_id<0:
            next
        xs, ys, zs = np.where(nmt.annotation.array==structure_id)
        lab = np.random.choice(np.arange(len(xs)), np.min([len(xs), 200]), replace=False)
        ax.scatter(zs[lab], xs[lab], ys[lab], label=cur_region, alpha=0.2)
    
    ax.legend()
    
    ax.scatter(615, 603, 679, marker="*")





.. parsed-literal::

    <mpl_toolkits.mplot3d.art3d.Path3DCollection at 0x7f4429864e80>




.. image:: output_4_1.png

