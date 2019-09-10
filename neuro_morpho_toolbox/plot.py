import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .swc import neuron
import colorlover as cl

#import chart_studio.plotly as py
import plotly.offline as po
import plotly.graph_objs as go

import seaborn as sns

# Global variables
u_views = ['Coronal', 'Horizontal', 'Sagittal']
u_color_by = ['SingleCell', 'Celltype', 'Subtype', 'cluster', 'nblast']

view_idx = dict(zip(u_views, [0, 1, 2]))
view_axis = dict(zip(u_views, ["X", "Y", "Z"]))

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


def get_group_colors(metadata, group_by="Celltype", palette="paired", return_str=False):
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

def swc_to_edges(swc):
    '''
    :param swc: a dataframe from the attribute of the neuron object (see ./swc.py neuron.swc).
    :return: a list of node coordinates and attributes for the convenience of plotting
    '''
    Xe = []
    Ye = []
    Ze = []
    Te = []
    Le = []
    for i, cur_child in enumerate(swc.index.tolist()):
        cur_parent = swc.loc[cur_child, "parent"]
        if cur_parent == -1:
            continue
        Xe = Xe + [swc.x[cur_child], swc.x[cur_parent], None]
        Ye = Ye + [swc.y[cur_child], swc.y[cur_parent], None]
        Ze = Ze + [swc.z[cur_child], swc.z[cur_parent], None]
        seg_length = ((swc.x[cur_child] - swc.x[cur_parent]) ** 2 +
                      (swc.y[cur_child] - swc.y[cur_parent]) ** 2 +
                      (swc.z[cur_child] - swc.z[cur_parent]) ** 2) ** 0.5
        Le = Le + [seg_length, 0, 0]
        if swc.type[cur_child] in [1, 2, 3, 4]:
            if swc.type[cur_parent] == 1:
                Te = Te + [swc.type[cur_child], 1, swc.type[cur_child]]
            else:
                Te = Te + [swc.type[cur_child]] * 3
        else:
            Te = Te + [0] * 3
    return [Xe, Ye, Ze, Te, Le]

def plot_swc_mpl(segment,
                 color='rgb(255, 0, 0)', view_by='Horizontal',
                 linewidth=1, alpha=1,
                 ax=None):
    assert view_by in u_views, " ".join((["option 'view_by' should be one of the following: "] + u_views))
    Xe, Ye, Ze, Te, Le = segment

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
    Xe, Ye, Ze, Te, Le = swc_to_edges(swc_name, metadata, segment_dict=segment_dict, flip=flip, z_size=z_size)
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


# def qualitative_scatter(x, y, c):
#     max_col = 3
#     subplot_w = 8
#     subplot_h = 8
#     feature_list = c.columns.tolist()
#     subplot_n = len(feature_list)
#     if subplot_n <= max_col:
#         n_col = subplot_n
#         n_row = 1
#     else:
#         n_col = max_col
#         n_row = int(subplot_n / max_col)
#         if (subplot_n % max_col) != 0:
#             n_row += 1
#     fig, ax = plt.subplots(n_row, n_col,
#                            figsize=(subplot_w * n_col,
#                                     subplot_h * n_row),
#                            squeeze=False
#                            )
#     ax = ax.reshape(-1)
#     df = pd.DataFrame({'Dim_1': x, 'Dim_2': y})
#     df = pd.concat([df, c.copy()], axis=1)
#     for i, cur_ax in enumerate(ax.tolist()[:subplot_n]):
#         feature_name = feature_list[i]
#         ct = df[feature_name].value_counts()
#         # Control length of legend
#         if len(ct) > 10:
#             collapsed_features = ct[7:].index.tolist() + ['unknown', "fiber tracts"]
#             df.loc[df[feature_name].isin(collapsed_features), feature_name] = "Others"
#
#         ct = df[feature_name].value_counts()
#         hue_order = ct.index.tolist()
#         if hue_order.count('Others') > 0:
#             #             hue_order.append(hue_order.pop(hue_order.index('Others')))
#             hue_order.pop(hue_order.index('Others'))
#
#         sns.scatterplot(x='Dim_1', y='Dim_2',
#                         data=df[df[feature_name] == 'Others'],
#                         c=(.5, .5, .5),
#                         alpha=0.25,
#                         #                         palette='Spectral',
#                         ax=cur_ax)
#         sns.scatterplot(x='Dim_1', y='Dim_2',
#                         hue=feature_name, hue_order=hue_order,
#                         data=df[df[feature_name] != 'Others'],
#                         #                         palette='Spectral',
#                         alpha=0.75,
#                         ax=cur_ax)
#     return fig


def qualitative_scatter(x, y, c, palette='Spectral', max_colors=10):
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
                        alpha=0.9,
                        size="is_background",
                        sizes=(41, 71),
                        style="is_background",
                        ax=cur_ax)
    return fig


def cell_in_map(neurons_dict, cell_list, metadata, ccf_annotation,
                view="Horizontal",
                margin=0.05, dpi=80, enlarge=1.5, alpha=0.5, ax=None,
                color="classical", flip_soma=True):
    assert view in u_views, " ".join((["option 'view_by' should be one of the following: "] + u_views))

    # Background image
    nda = np.empty([0,0])
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

    single_cell_color_dict = get_singlecell_colors(cell_list, return_str=False)
    celltype_color_dict = get_group_colors(metadata=metadata, group_by="Celltype", palette="paired", return_str=False)
    cluster_color_dict = get_group_colors(metadata=metadata, group_by="cluster", palette="paired", return_str=False)
    for cellname in cell_list:
        Xe, Ye, Ze, Te, Le = swc_to_edges(neurons_dict[cellname].swc)
        # axis_name = view_axis[view]
        tp = pd.DataFrame(columns=['heng', 'zong', 'Te'])
        if view.lower() == "coronal":
            tp = pd.DataFrame({'heng': Ze, 'zong': Ye, 'Te': Te})
        if view.lower() == "horizontal":
            tp = pd.DataFrame({'heng': Ze, 'zong': Xe, 'Te': Te})
        if view.lower() == "sagittal":
            tp = pd.DataFrame({'heng': Xe, 'zong': Ye, 'Te': Te})
            flip_soma = False

        soma_color = single_cell_color_dict[cellname]
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
            soma_color = celltype_color_dict[metadata.loc[cellname, "Celltype"]]
            ax.plot(tp.heng, tp.zong,
                    c=celltype_color_dict[metadata.loc[cellname, "Celltype"]],
                    linewidth=linewidth, alpha=alpha)
        if color.lower() == "cluster":
            soma_color = cluster_color_dict[metadata.loc[cellname, "cluster"]]
            ax.plot(tp.heng, tp.zong,
                    c=cluster_color_dict[metadata.loc[cellname, "cluster"]],
                    linewidth=linewidth, alpha=alpha)
        tp = tp[(tp["Te"] == 1)]
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
    return

