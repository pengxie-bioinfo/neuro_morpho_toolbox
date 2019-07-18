import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .swc import neuron

import chart_studio.plotly as py
import plotly.offline as po
import plotly.graph_objs as go

import seaborn as sns

# Global variables
u_views = ['Coronal', 'Horizontal', 'Sagittal']
u_color_by = ['SingleCell', 'Celltype', 'Subtype', 'cluster', 'nblast']

view_idx = dict(zip(u_views, [0, 1, 2]))
view_axis = dict(zip(u_views, ["X", "Y", "Z"]))


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

# Plotting utilities
def rgb_to_list(rgb_str):
    rgb_str = rgb_str.replace(" ", "")
    tp = rgb_str.replace("rgb(","").replace("rgba(","").replace(")","")
    res = [float(i)/255 for i in tp.split(",")]
    return res


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


def qualitative_scatter(x, y, c):
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
    for i, cur_ax in enumerate(ax.tolist()[:subplot_n]):
        feature_name = feature_list[i]
        ct = df[feature_name].value_counts()
        # Control length of legend
        if len(ct) > 10:
            collapsed_features = ct[12:].index.tolist() + ['unknown', "fiber tracts"]
            df.loc[df[feature_name].isin(collapsed_features), feature_name] = "Others"

        ct = df[feature_name].value_counts()
        hue_order = ct.index.tolist()
        if hue_order.count('Others') > 0:
            #             hue_order.append(hue_order.pop(hue_order.index('Others')))
            hue_order.pop(hue_order.index('Others'))

        sns.scatterplot(x='Dim_1', y='Dim_2',
                        data=df[df[feature_name] == 'Others'],
                        c=(.5, .5, .5),
                        alpha=0.25,
                        #                         palette='Spectral',
                        ax=cur_ax)
        sns.scatterplot(x='Dim_1', y='Dim_2',
                        hue=feature_name, hue_order=hue_order,
                        data=df[df[feature_name] != 'Others'],
                        #                         palette='Spectral',
                        alpha=0.75,
                        ax=cur_ax)
    return fig
