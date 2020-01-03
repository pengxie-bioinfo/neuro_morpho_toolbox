import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .swc import neuron
import colorlover as cl
import time
#import chart_studio.plotly as py
import plotly.offline as po
import plotly.graph_objs as go
import neuro_morpho_toolbox as nmt
import seaborn as sns

# Global variables
u_views = ['Coronal', 'Horizontal', 'Sagittal']
u_color_by = ['SingleCell', 'CellType', 'Subtype', 'Cluster', 'nblast']


view_idx = dict(zip(u_views, [0, 1, 2]))
view_axis = dict(zip(u_views, ["X", "Y", "Z"]))


# 6-Nearest Neighbor Atlas with selected CCF ID on
ccf_Contour = np.multiply(nmt.ccfArray,nmt.Contour01)

####################################################################################
# Color settings
####################################################################################
bupu = cl.scales['9']['seq']['BuPu']
greens = cl.scales['9']['seq']['Greens']
set2 = cl.scales['7']['qual']['Set2']
spectral = cl.scales['9']['div']['Spectral']
paired = cl.scales['10']['qual']['Paired']
mpl_colors = []
for i in range(9):
    tp = []
    for j in list(mpl.colors.to_rgb("C" + str(i))):
        tp.append(str(int(j * 255)))
    tp = ", ".join(tp)
    tp = "rgb(" + tp + ")"
    mpl_colors.append(tp)
my_palette_dict = {"bupu":bupu,
                   "greens":greens,
                   "set2":set2,
                   "spectral":spectral,
                   "paired":paired,
                   "matplotlib":mpl_colors
                   }

def rgb_to_list(rgb_str):
    tp = rgb_str.replace("rgb(", "").replace("rgba(", "").replace(")", "")
    res = [float(i) / 255 for i in tp.split(", ")]
    return res


def get_group_colors(metadata, group_by="CellType", palette="paired", return_str=False):
    assert group_by in metadata.columns.tolist(), "Invalid group_by value."
    assert palette in list(my_palette_dict.keys()), "Invalid palette name."
    u_groups = sorted(list(set(metadata[group_by])))
    color_list = cl.to_rgb(cl.interp(my_palette_dict[palette], len(u_groups)))
    if not return_str:
        color_list = [rgb_to_list(i) for i in color_list]
    group_colors = dict(zip(u_groups, color_list))
    if "Others" in u_groups:
        if not return_str:
            group_colors["Others"] = rgb_to_list('rgb(128, 128, 128)')
        else:
            group_colors["Others"] = 'rgb(128, 128, 128)'
    return group_colors

def get_singlecell_colors(cell_list, palette="paired", return_str=False):
    u_cells = list(set(cell_list))
    color_list = cl.to_rgb(cl.interp(my_palette_dict[palette], len(u_cells)))
    if not return_str:
        color_list = [rgb_to_list(i) for i in color_list]
    group_colors = dict(zip(u_cells, color_list))
    return group_colors

# SWC plotting
def get_layout(axis_x, axis_y, range_x, range_y, x_expand_ratio=1.1):
    '''
    Get layout for plotly setting
    :param axis_x:
    :param axis_y:
    :param range_x:
    :param range_y:
    :param x_expand_ratio:
    :return:
    '''
    layout = {"xaxis": dict(title=axis_x, range=[range_x[0], range_x[1] * x_expand_ratio]),
              "yaxis": dict(title=axis_y, range=range_y),
              "width": range_x[1] / 20 * x_expand_ratio,
              "height": range_y[1] / 20,
              "margin": dict(autoexpand=False)
              }
    return layout


def soma_to_edges(swc):
    '''
    :param swc: a dataframe from the attribute of the neuron object (see ./swc.py neuron.swc).
    :return: a list of node coordinates and attributes for the convenience of plotting
    '''
    Xe = []
    Ye = []
    Ze = []
    Te = []
    if swc[swc.type==1].shape[0]>0:
        somaDF = swc[swc.type==1].copy()
        Xe = Xe + [somaDF.x.iloc[0]]
        Ye = Ye + [somaDF.y.iloc[0]]
        Ze = Ze + [somaDF.z.iloc[0]]
        Te = Te + [1]
    return [Xe, Ye, Ze, Te]


def swc_to_edges(swc, keep_invalid=True):
    '''
    :param swc: a dataframe from the attribute of the neuron object (see ./swc.py neuron.swc).
    :return: a list of node coordinates and attributes for the convenience of plotting
    '''

    # 1. Get all edges
    all_nodes = swc.index.tolist()
    children = swc.index[swc.parent.isin(all_nodes)].tolist()
    N = len(children)
    parents = swc.loc[children, "parent"].tolist()
    cuts = [None] * N

    # 2. Create a dataframe with all edges
    # each edge is represented as 3 consecutive rows: [child, parent, empty]
    res = pd.DataFrame(columns=['x', 'y', 'z', 'type'],
                       index=range(N * 3)
                       )
    children_id = np.arange(N) * 3
    parents_id = np.arange(N) * 3 + 1
    cuts_id = np.arange(N) * 3 + 2

    # 2.1. Location columns

    for cur_col in ['x', 'y', 'z']:
        res.loc[children_id, cur_col] = swc.loc[children, cur_col].tolist()
        res.loc[parents_id, cur_col] = swc.loc[parents, cur_col].tolist()
    res.loc[cuts_id, ['x', 'y', 'z']] = None

    # 2.2 Type column
    res.loc[children_id, 'type'] = swc.loc[children, 'type'].tolist()
    res.loc[parents_id, 'type'] = swc.loc[children, 'type'].tolist()  # Use child type to represent a segment
    res.loc[cuts_id, 'type'] = swc.loc[children, 'type'].tolist()
    # Soma type
    soma_list = swc[swc.type == 1].index.tolist()
    if len(soma_list) > 0:
        i_soma = [parents_id[parents.index(i)] for i in soma_list if i in parents]
        res.loc[i_soma, 'type'] = 1
    # Invalid types
    res.loc[~res.type.isin([1, 2, 3, 4]), "type"] = 0
    if not keep_invalid:
        res = res[res.type!=0]
    # Output
    Xe = res.x.tolist()
    Ye = res.y.tolist()
    Ze = res.z.tolist()
    Te = res.type.tolist()

    return [Xe, Ye, Ze, Te]


def plot_swc_mpl(segment,
                 color='rgb(255, 0, 0)', view_by='Horizontal',
                 linewidth=1, alpha=1,
                 ax=None):
    assert view_by in u_views, " ".join((["option 'view_by' should be one of the following: "] + u_views))
    Xe, Ye, Ze, Te = segment

    if view_by == "Coronal":
        tp = pd.DataFrame({'heng': Ze, 'zong': Ye, 'Te': Te})
    if view_by == "Horizontal":
        tp = pd.DataFrame({'heng': Ze, 'zong': Xe, 'Te': Te})
    if view_by == "Sagittal":
        tp = pd.DataFrame({'heng': Ye, 'zong': Xe, 'Te': Te})

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if color == "classical":
        ax.plot(tp.heng[tp.Te == 2], tp.zong[tp.Te == 2], c='r', linewidth=linewidth, alpha=alpha)
        ax.plot(tp.heng[tp.Te == 3], tp.zong[tp.Te == 3], c='b', linewidth=linewidth, alpha=alpha)
        ax.plot(tp.heng[tp.Te == 4], tp.zong[tp.Te == 4], c='magenta', linewidth=linewidth, alpha=alpha)
    if color.startswith('rgb'):
        ax.plot(tp.heng, tp.zong,
                c=rgb_to_list(color),
                linewidth=linewidth, alpha=alpha)
    tp = tp[(tp["Te"] == 1)]
    ax.scatter(tp.heng, tp.zong,
               c='black',
               marker="o",
               s=30)
    return


def plot_swc_plotly(swc_name, metadata, segment_dict={}, flip=False, z_size=None,
             color='rgb(255,0,0)', show_plot=True, view_by='Horizontal', ranges=None):
    assert view_by in u_views, " ".join((["option 'view_by' should be one of the following: "] + u_views))
    Xe, Ye, Ze, Te = swc_to_edges(swc_name, metadata, segment_dict=segment_dict, flip=flip, z_size=z_size)
    # if ranges is not None:
    #     xrange, yrange, zrange = ranges

    if view_by == 'Horizontal':
        plot_x, plot_y = [Ze, Xe]
        axis_x, axis_y = ["Z-axis", "X-axis"]
        # range_x, range_y = [zrange, xrange]
    elif view_by == 'Coronal':
        plot_x, plot_y = [Ze, Ye]
        axis_x, axis_y = ["Z-axis", "Y-axis"]
        # range_x, range_y = [zrange, yrange]
    elif view_by == 'Sagittal':
        plot_x, plot_y = [Xe, Ye]
        axis_x, axis_y = ["X-axis", "Y-axis"]
        # range_x, range_y = [xrange, yrange]
    lines = go.Scatter(x=plot_x,
                       y=plot_y,
                       mode='lines',
                       line=dict(color=color, width=1),  # TODO: change line color for types
                       hoverinfo='none',
                       name="_".join(swc_name.split("_")[-2:]),
                       opacity=0.8
                       )
    if show_plot:
        po.iplot({"data": [lines],
                  # "layout": get_layout(axis_x, axis_y, range_x, range_y, 1.1)
                  })
    return lines


def cell_in_map(neurons_dict, cell_list, metadata, ccf_annotation,
                view="Horizontal",
                margin=0.05, dpi=80, enlarge=1.5, alpha=0.5, ax=None,
                color="classical", flip_soma=True):
    assert view in u_views, " ".join((["option 'view_by' should be one of the following: "] + u_views))

    # Background image
    nda = np.empty([0, 0])
    xspace = 0
    yspace = 0
    if view.lower() == "coronal":
        nda = (np.max(ccf_annotation.array, axis=0) > 0)  # 3D -> 2D projection
        xspace = ccf_annotation.space['z']
        yspace = ccf_annotation.space['y']
    if view.lower() == "horizontal":
        nda = (np.max(ccf_annotation.array, axis=1) > 0)
        xspace = ccf_annotation.space['z']
        yspace = ccf_annotation.space['x']
    if view.lower() == "sagittal":
        nda = (np.max(ccf_annotation.array, axis=2) > 0).transpose()
        xspace = ccf_annotation.space['y']
        yspace = ccf_annotation.space['x']

    xsize = nda.shape[1]
    ysize = nda.shape[0]

    # Figure settings
    if ax is None:
        figsize = (1 + margin) * xsize * enlarge / dpi, (1 + margin) * ysize * enlarge / dpi
        fig = plt.figure(figsize=figsize, dpi=dpi)
        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    extent = (0, xsize * xspace, ysize * yspace, 0)
    ax.imshow(nda, cmap="Greys", alpha=0.2, extent=extent)

    # Plot cells
    linewidth = 0.7
    alpha = 0.7
    if color.lower() == "single_cell" or color.lower() == "soma":
        single_cell_color_dict = get_singlecell_colors(cell_list, return_str=False)
    if color.lower() == "celltype":
        celltype_color_dict = get_group_colors(metadata=metadata, group_by="CellType", palette="spectral", return_str=False)
    if color.lower() == "cluster" or color.lower() == "majorsoma":
        cluster_color_dict = get_group_colors(metadata=metadata, group_by="Cluster", palette="paired", return_str=False)

    # Option 1: Soma-only mode
    if color.lower() == "soma" or color.lower() == "majorsoma":
        print('Illustrating soma locations inside a brain from ' + view.lower() + ' view:')
        for cellname in cell_list:
            Xe, Ye, Ze, Te = soma_to_edges(neurons_dict[cellname].swc)
            # axis_name = view_axis[view]
            tp = pd.DataFrame(columns=['heng', 'zong', 'Te'])
            if view.lower() == "coronal":
                tp = pd.DataFrame({'heng': Ze, 'zong': Ye, 'Te': Te})
            if view.lower() == "horizontal":
                tp = pd.DataFrame({'heng': Ze, 'zong': Xe, 'Te': Te})
            if view.lower() == "sagittal":
                tp = pd.DataFrame({'heng': Xe, 'zong': Ye, 'Te': Te})
                flip_soma = False
            if color.lower() == "soma":
                soma_color = single_cell_color_dict[cellname]
            else:
                soma_color = cluster_color_dict[metadata.loc[cellname, "Cluster"]]
            if flip_soma:
                ax.scatter(xsize * xspace - tp.heng[tp["Te"] == 1].iloc[0],
                           tp.zong[tp["Te"] == 1].iloc[0],
                           c=[soma_color],
                           marker="o",
                           s=30)
            else:
                ax.scatter(tp.heng[tp["Te"] == 1].iloc[0],
                           tp.zong[tp["Te"] == 1].iloc[0],
                           c=[soma_color],
                           marker="o",
                           s=30)
        return

    # Option 2: swc mode
    start_sum = time.time()
    i_p = 0
    for cellname in cell_list:
        # TBI: replace by plot_swc_mpl
        start_sub = time.time()
        i_p = i_p + 1
        # print("Processing progress: %.2f" % (i_p / len(cell_list)))
        Xe, Ye, Ze, Te = swc_to_edges(neurons_dict[cellname].swc)
        tp = pd.DataFrame(columns=['heng', 'zong', 'Te'])
        if view.lower() == "coronal":
            tp = pd.DataFrame({'heng': Ze, 'zong': Ye, 'Te': Te})
        if view.lower() == "horizontal":
            tp = pd.DataFrame({'heng': Ze, 'zong': Xe, 'Te': Te})
        if view.lower() == "sagittal":
            tp = pd.DataFrame({'heng': Xe, 'zong': Ye, 'Te': Te})
            flip_soma = False

        if color.lower() == "classical":
            ax.plot(tp.heng[tp.Te == 2], tp.zong[tp.Te == 2], c='r', linewidth=linewidth, alpha=alpha)
            ax.plot(tp.heng[tp.Te == 3], tp.zong[tp.Te == 3], c='b', linewidth=linewidth, alpha=alpha)
            ax.plot(tp.heng[tp.Te == 4], tp.zong[tp.Te == 4], c='magenta', linewidth=linewidth, alpha=alpha)
            soma_color = 'black'
        if color.lower() == "single_cell":
            soma_color = single_cell_color_dict[cellname]
            ax.plot(tp.heng, tp.zong,
                    c=single_cell_color_dict[cellname],
                    linewidth=linewidth, alpha=alpha)
        if color.lower() == "celltype":
            soma_color = celltype_color_dict[metadata.loc[cellname, "CellType"]]
            ax.plot(tp.heng, tp.zong,
                    c=celltype_color_dict[metadata.loc[cellname, "CellType"]],
                    linewidth=linewidth, alpha=alpha)
        if color.lower() == "cluster":
            soma_color = cluster_color_dict[metadata.loc[cellname, "Cluster"]]
            ax.plot(tp.heng, tp.zong,
                    c=cluster_color_dict[metadata.loc[cellname, "Cluster"]],
                    linewidth=linewidth, alpha=alpha)
        tp = tp[(tp["Te"] == 1)]

        # Show soma location
        if flip_soma:
            ax.scatter(xsize * xspace - tp.heng[tp["Te"] == 1].iloc[0],
                       tp.zong[tp["Te"] == 1].iloc[0],
                       c=[soma_color],
                       marker="*",
                       s=30)
        else:
            ax.scatter(tp.heng[tp["Te"] == 1].iloc[0],
                       tp.zong[tp["Te"] == 1].iloc[0],
                       c=[soma_color],
                       marker="*",
                       s=30)
        end_sub = time.time()
        # print("Single cell's loading time: %.2f" % (end_sub - start_sub))
    end_sum = time.time()
    print("Total loading time: %.2f" % (end_sum - start_sum))
    return

def quantitative_scatter(x, y, c, cmap='bwr', alpha=0.75, s=5):
    max_col = 3
    subplot_w = 6
    subplot_h = 6
    feature_list = c.columns.tolist()
    subplot_n = len(feature_list)
    if subplot_n <= max_col:
        n_col = subplot_n
        n_row = 1
    else:
        n_col = max_col
        n_row = int(subplot_n / max_col)
        if (subplot_n % max_col) != 0:
            n_row += 1
    fig, ax = plt.subplots(n_row, n_col,
                           figsize=(subplot_w * n_col,
                                    subplot_h * n_row),
                           squeeze=False
                           )
    ax = ax.reshape(-1)
    for i, cur_ax in enumerate(ax.tolist()[:subplot_n]):
        feature_name = c.columns.tolist()[i]
        cur_ax.scatter(x, y,
                       c=(.5, .5, .5),
                       s=s,
                       alpha=0.5)
        P = cur_ax.scatter(x, y,
                           c=c[feature_name],
                           cmap=cmap,
                           s=s,
                           vmax=np.percentile(c[feature_name], q=95),
                           vmin=np.percentile(c[feature_name], q=5),
                           alpha=alpha)
        cur_ax.set_xlabel("Dim1")
        cur_ax.set_ylabel("Dim2")
        cur_ax.set_title(feature_name)
        # Creating color bar
        divider = make_axes_locatable(cur_ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(P, cax)
    return fig


def qualitative_scatter(x, y, c, palette='Spectral', max_colors=25):
    max_col = 3
    subplot_w = 8
    subplot_h = 8
    feature_list = c.columns.tolist()
    subplot_n = len(feature_list)
    if subplot_n <= max_col:
        n_col = subplot_n
        n_row = 1
    else:
        n_col = max_col
        n_row = int(subplot_n / max_col)
        if (subplot_n % max_col) != 0:
            n_row += 1
    fig, ax = plt.subplots(n_row, n_col,
                           figsize=(subplot_w * n_col,
                                    subplot_h * n_row),
                           squeeze=False
                           )
    ax = ax.reshape(-1)
    df = pd.DataFrame({'Dim_1': x, 'Dim_2': y})
    df = pd.concat([df, c.copy()], axis=1)
    df['is_background'] = 'N'
    for i, cur_ax in enumerate(ax.tolist()[:subplot_n]):
        feature_name = feature_list[i]
        ct = df[feature_name].value_counts()
        # Control length of legend
        if len(ct) > max_colors:
            collapsed_features = ct[max_colors:].index.tolist() + ['unknown', "fiber tracts"]
            df.loc[df[feature_name].isin(collapsed_features), feature_name] = "NA"
            df.loc[df[feature_name].isin(["Others", "NA"]), "is_background"] = "Y"

        # background color
        if type(palette) == dict:
            palette['Others'] = (.5, .5, .5)
            palette['NA'] = (.5, .5, .5)

        ct = df[feature_name].value_counts()
        hue_order = ct.index.tolist()
        sns.scatterplot(x='Dim_1', y='Dim_2',
                        hue=feature_name,
                        hue_order = hue_order,
                        data=df,
                        palette=palette,
                        alpha=0.8,
                        size="is_background",
                        sizes={'N':60, 'Y':30},
                        style="is_background",
                        linewidth=0,
                        ax=cur_ax)
    return fig

def border_line(view, position, regions=None, ax=None, bkground_ON = False):
    margin=0.05
    dpi=80
    enlarge=1.5
    alpha=0.5
    ccf_annotation = nmt.annotation
    # Background image
    nda = np.empty([0, 0])
    xspace = 0
    yspace = 0
    if view.lower() == "coronal":
        nda = (np.max(ccf_annotation.array, axis=0) > 0)  # 3D -> 2D projection
        xspace = ccf_annotation.space['z']
        yspace = ccf_annotation.space['y']
    if view.lower() == "horizontal":
        nda = (np.max(ccf_annotation.array, axis=1) > 0)
        xspace = ccf_annotation.space['z']
        yspace = ccf_annotation.space['x']
    if view.lower() == "sagittal":
        nda = (np.max(ccf_annotation.array, axis=2) > 0).transpose()
        xspace = ccf_annotation.space['y']
        yspace = ccf_annotation.space['x']

    xsize = nda.shape[1]
    ysize = nda.shape[0]

    # Figure settings
    if ax is None:
        figsize = (1 + margin) * xsize * enlarge / dpi, (1 + margin) * ysize * enlarge / dpi
        fig = plt.figure(figsize=figsize, dpi=dpi)
        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    extent = (0, xsize * xspace, ysize * yspace, 0)
    if bkground_ON:
        ax.imshow(nda, cmap="Greys", alpha=0.0, extent=extent)
    # else:
    #     ax.imshow(nda, cmap="Greys", alpha=1.0, extent=extent)
    if regions != None:
        x_range = np.array([])
        y_range = np.array([])
        if view.lower() == "coronal":    #x   z,y
            assert position < nmt.annotation.array.shape[0],"Input position must within the brain region"
            for iter_Region in regions:
                if type(iter_Region) == str:
                    x_range = np.append(x_range,np.where(ccf_Contour[position,:,:] == nmt.bs.name_to_id(iter_Region))[1])
                    y_range = np.append(y_range,np.where(ccf_Contour[position,:,:] == nmt.bs.name_to_id(iter_Region))[0])
        if view.lower() == "horizontal": #y z,x
            assert position < nmt.annotation.array.shape[1],"Input position must within the brain region"
            for iter_Region in regions:
                if type(iter_Region) == str:
                    x_range = np.append(x_range,np.where(ccf_Contour[:,position,:] == nmt.bs.name_to_id(iter_Region))[1])
                    y_range = np.append(y_range,np.where(ccf_Contour[:,position,:] == nmt.bs.name_to_id(iter_Region))[0])
        if view.lower() == "sagittal":   #z  y,x
            assert position < nmt.annotation.array.shape[2],"Input position must within the brain region"
            for iter_Region in regions:
                if type(iter_Region) == str:
                    x_range = np.append(x_range,np.where(ccf_Contour[:,:,position] == nmt.bs.name_to_id(iter_Region))[1])
                    y_range = np.append(y_range,np.where(ccf_Contour[:,:,position] == nmt.bs.name_to_id(iter_Region))[0])
        ax.scatter(xspace * x_range, yspace* y_range, marker="o",s=3)
    return