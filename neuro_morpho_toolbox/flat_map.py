import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
import umap
from sklearn import metrics

from .utilities import cart2pol_3d
from .brain_structure import *
from neuro_morpho_toolbox import annotation
from neuro_morpho_toolbox import bs

import matplotlib.pyplot as plt
import seaborn as sns

midline = int(annotation.size['z'] / 2)
space = annotation.space['x']

'''
Expand the cortex as a flat 2D map. Expansion is performed in a coronal-slice manner.
The map is annotated by brain regions. 
For each slice, we define a set of anchor points, where inner layer points should project to.
'''
def my_normalize(x):
    return x / np.sqrt(x[0]**2+x[1]**2)

def my_projection(x,y):
    return np.dot(x, y)

###################################################### Hard code!
import pickle
[Contour_layer] = pickle.load(open('/Users/pengxie/Documents/Research/Thalamus_fullMorpho/ipython/cortical_layer_contour.pickle', 'rb'))
Contour_layer = Contour_layer[:,:,:midline]
[cortex_layer_array] = pickle.load(open('/Users/pengxie/Documents/Research/Thalamus_fullMorpho/ipython/cortical_layer_array.pickle', 'rb'))
cortex_layer_array = cortex_layer_array[:,:,:midline]
[cortex_region_array] = pickle.load(open('/Users/pengxie/Documents/Research/Thalamus_fullMorpho/ipython/cortical_regions_array.pickle', 'rb'))
cortex_region_array = cortex_region_array[:,:,:midline]

layer_list = ['L1', 'L2/3', 'L4', 'L5', 'L6a', 'L6b']
layer_dict = dict(zip([1,2,3,4,5,6], layer_list))
layer_dict_reverse = dict(zip(layer_list, [1,2,3,4,5,6]))
cortex_array_dict = {'region':cortex_region_array, 'layer':cortex_layer_array}

######################################################

def get_shell(x_position, plot=False):
    '''

    :param x_position: float, in the unit of um
    :param plot: boolean, whether to plot the current shell
    :return: shell dataframe, columns=['x', 'y', 'parent', 'x_expand']
    '''
    # 1. Extract shell
    x = Contour_layer[int(x_position / space), :, :]
    where = np.where(x > 0)
    if len(where[0]) == 0:
        return None
    where = pd.DataFrame({
        'x': where[1],
        'y': where[0],
        'is_shell': False
    })
    for i in range(240):
        # Scan vertically
        tp = where[(where.x == i)]
        if len(tp > 0):
            cur_ind = tp.sort_values(['y']).index[0]
            where.loc[cur_ind, 'is_shell'] = True
    for i in range(240):
        # Scan horizontally
        tp = where[(where.y == i)]
        if len(tp > 0):
            cur_ind = tp.sort_values(['x']).index[0]
            where.loc[cur_ind, 'is_shell'] = True
    # 2. connect points on the shell
    shell = where[where.is_shell].sort_values(['x']).copy()[['x', 'y']]
    shell.index = range(len(shell))
    distance = pd.DataFrame(metrics.pairwise_distances(shell[['x', 'y']]),
                            index=shell.index,
                            columns=shell.index
                            )
    start = shell.sort_values(['y'], ascending=False).index.tolist()[0]
    end = shell.sort_values(['x'], ascending=False).index.tolist()[0]
    print("Start: %d, %.1f, %.1f" % (start, shell.loc[start, 'x'], shell.loc[start, 'y']))
    shell['parent'] = -1
    cur_node = start
    node_list = [start]
    while cur_node != end:
        prev_node = cur_node
        tp = distance[cur_node].sort_values()[1:]
        tp_list = [i for i in tp.index.tolist() if not i in node_list]
        for i in tp_list:
            if shell.loc[i, 'parent'] == -1:  # Found child node
                shell.loc[i, 'parent'] = cur_node
                cur_node = i
                node_list = node_list + [cur_node]
                break
        if cur_node == prev_node:
            # No neighbor found
            print("Cannot find neighbor for %d." % (cur_node))
            break
    shell = shell.loc[node_list]

    # 3. add a column for anchors to show the coordinate in the expanded map
    shell['x_expand'] = -9999
    start = shell.index[shell.parent == -1][0]
    shell.loc[start, 'x_expand'] = 0
    cur_node = start
    while len(shell[shell.parent == cur_node]) == 1:
        prev_node = cur_node
        cur_node = shell.index[shell.parent == cur_node][0]
        cur_distance = shell.loc[cur_node, ['x', 'y']] - shell.loc[prev_node, ['x', 'y']]
        cur_distance = np.sqrt(np.sum(np.square(cur_distance)))
        shell.loc[cur_node, 'x_expand'] = shell.loc[prev_node, 'x_expand'] + cur_distance

    # Alignment: find the the middle-most shell and set its cooridinate as 0
    mid_x_expand = shell.x_expand[shell.x == np.max(shell.x)].iloc[0]
    shell['x_expand'] = shell['x_expand'] - mid_x_expand

    # 4. add a column for region names
    shell['region'] = get_region_from_shell(shell, x_position)

    # Optinal: plot data
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].scatter(where.x, where.y)
        ax[0].scatter(where.x[where.is_shell],
                      where.y[where.is_shell]
                      )
        ax[0].set_ylim(240, 0)

        ax[1].plot(shell.x, shell.y)
        ax[1].scatter(shell.loc[start].x, shell.loc[start].y, c='r')
        ax[1].scatter(shell.loc[end].x, shell.loc[end].y, c='g')
        ax[1].set_ylim(240, 0)
    return shell


def get_normal_vector(shell, cur_node, n_neighbors):
    distance = pd.DataFrame(metrics.pairwise_distances(shell[['x', 'y']]),
                            index=shell.index,
                            columns=shell.index
                            )
    while True:
        tp = distance[cur_node].sort_values()[:n_neighbors].index.tolist()
        cur_neighbors = shell.loc[tp].copy()[['x', 'y']]

        pca = PCA(n_components=2)
        pca.fit(cur_neighbors)
        pc_1 = pca.components_[0, :]
        pc_2 = pca.components_[1, :]
        if ((pc_2[0] != 0) | (pc_2[1] != 0)):
            break
        n_neighbors = n_neighbors + 2
    return my_normalize(pc_2)

# For testing purpose
def plot_anchor(shell, anchor, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 8))
    anchor_list = anchor.index.tolist()
    ax.plot(shell.x, shell.y)
    ax.scatter(shell.loc[anchor_list].x, shell.loc[anchor_list].y)
    for cur_node in anchor_list:
        ax.plot(np.array([0, anchor.loc[cur_node, 'vx'] * 10]) + shell.loc[cur_node, 'x'],
                np.array([0, anchor.loc[cur_node, 'vy'] * 10]) + shell.loc[cur_node, 'y']
                )
    ax.set_xlim(0, midline)
    ax.set_ylim(240, 0)
    return ax
def plot_cortex(x_position, color='region', ax=None):
    y = cortex_array_dict[color]
    y = y[int(x_position/space), :, :]
    pixels = np.where(y>0)
    pixels = np.array([pixels[1], pixels[0]])
    pixels = np.transpose(pixels)
    cur_slice = pd.DataFrame(pixels, columns=['x', 'y'], dtype=np.int64)
    cur_slice['id'] = y[cur_slice.y, cur_slice.x]
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(7.2,8))
    if color=='region':
        cur_slice['name'] = [bs.id_to_name(i) for i in cur_slice.id.tolist()]
        sns.scatterplot(x='x', y='y', hue='name', data=cur_slice, ax=ax)
    if color=='layer':
        cur_slice['name'] = [layer_dict[i] for i in cur_slice.id.tolist()]
        cur_slice.name = pd.Categorical(cur_slice.name, categories=layer_list)
        sns.scatterplot(x='x', y='y',
                    hue='name', hue_order=layer_list,
                    data=cur_slice, ax=ax)
    ax.set_xlim(0, midline)
    ax.set_ylim(240, 0)
    ax.legend(loc='lower right')
    return ax

def get_anchor(shell, x_position, step=5, n_neighbors=20, plot=False):
    '''
    Get anchor points to estimate local normal direction, for the purpose of 2D surface projection.

    Parameters:
    shell: a dataframe. see 'get_shell' for format info
    step: get anchor points every 'step' nodes
    n_neighbors: number of neighbors for estimating normal direction

    '''

    node_list = shell.index.tolist()
    end = shell.index.tolist()[-1]
    # 1. Find anchor points
    anchor_list = [ind for i, ind in enumerate(node_list) if ((i % step == 0) | (i == end))]
    anchor = shell.loc[anchor_list].copy()
    anchor['vx'] = 0  # normal vectors
    anchor['vy'] = 0

    for cur_node in anchor_list:
        pc_2 = get_normal_vector(shell, cur_node, n_neighbors)
        # invert the vector if it's pointing to the outside of the cortex
        vx = int((pc_2[0] * 2 + shell.loc[cur_node, 'x']))
        vy = int((pc_2[1] * 2 + shell.loc[cur_node, 'y']))
        if ((vx < 0) | (vy < 0) | (vx >= cortex_region_array.shape[2]) | (vy >= cortex_region_array.shape[1])):
            pc_2 = pc_2 * (-1)
        elif annotation.array[int(x_position / space), vy, vx] == 0:
            pc_2 = pc_2 * (-1)

        anchor.loc[cur_node, 'vx'] = pc_2[0]
        anchor.loc[cur_node, 'vy'] = pc_2[1]
    if plot:
        plot_anchor(shell, anchor)
    #         fig, ax = plt.subplots(1,1,figsize=(8,8))
    #         ax.plot(shell.x, shell.y)
    #         ax.scatter(shell.loc[anchor_list].x, shell.loc[anchor_list].y)
    #         for cur_node in anchor_list:
    #             ax.plot(np.array([0, anchor.loc[cur_node, 'vx']*10]) + shell.loc[cur_node, 'x'],
    #                     np.array([0, anchor.loc[cur_node, 'vy']*10]) + shell.loc[cur_node, 'y']
    #                    )
    #         ax.set_ylim(240,0)
    return anchor

def get_projection_feild(x_position, anchor, plot = False):
    y = cortex_region_array[int(x_position / space), :, :midline]
    # Find anchor for each pixel
    pixels = np.where(y>0)
    pixels = np.array([pixels[1], pixels[0]])
    pixels = np.transpose(pixels)

    # 1. Distance from pixel to anchors
    distance = metrics.pairwise_distances(X=pixels, Y=np.array(anchor[['x', 'y']]))
    # 2. Anchor assignment
    df = pd.DataFrame(pixels,
                      columns=['x', 'y'],
                      index=[str(pixels[i,0])+"_"+str(pixels[i,1]) for i in range(len(pixels))]
                     )
    df['anchor'] = [anchor.index[i] for i in list(np.argmin(distance, axis=1))]

    # For checking purpose
    if plot:
        df_plot = df.copy()
        df_plot['anchor'] = list(np.argmin(distance, axis=1))
        sns.relplot(x='x', y='y',
                    hue='anchor',
                    data=df_plot,
                   )
    return df

class slice:
    def __init__(self, x_position, x_thickness, anchor_step=5):
        self.x_position = x_position
        self.x_thickness = x_thickness
        self.shell = get_shell(x_position, plot=False)
        if self.shell is None:
            self.anchor = None
            self.pf = None
            return
        self.anchor = get_anchor(self.shell,
                                 x_position=x_position,
                                 step=anchor_step,
                                 n_neighbors=20,
                                 plot=False
                                 )
        x_position_scaled = int(x_position / space)
        self.shell.index = ["_".join([str(x_position_scaled),
                                       str(self.shell.loc[i, 'y']),
                                       str(self.shell.loc[i, 'x'])]) for i in self.shell.index.tolist()]
        self.anchor.index = ["_".join([str(x_position_scaled),
                                       str(self.anchor.loc[i, 'y']),
                                       str(self.anchor.loc[i, 'x'])]) for i in self.anchor.index.tolist()]

        self.pf = get_projection_feild(x_position, self.anchor, False)
        return

def get_boundary_from_shell(shell):
    res = pd.DataFrame(~shell.index.isin(shell['parent'].value_counts().index), index=shell.index)[0]
    res[shell.index[shell['parent']==-1]] = True
    return res

def get_region_from_shell(shell, cur_x_slice):
    x_ccf_list = [int(cur_x_slice/space)] * len(shell)
    y_ccf_list = shell.loc[shell.index, 'y'].tolist()
    z_ccf_list = shell.loc[shell.index, 'x'].tolist()
    region_list = list(cortex_region_array[x_ccf_list, y_ccf_list, z_ccf_list])
    region_list = [bs.id_to_name(i) for i in region_list]
#     res = pd.DataFrame({'x_ccf':x_ccf_list,
#                         'y_ccf':y_ccf_list,
#                         'z_ccf':z_ccf_list,
#                         'region':region_list
#                        })
    return region_list

class slice_set:
    def __init__(self, x_thickness, start=None, end=None, anchor_step=5):
        # 1. Basic information
        self.x_thickness = x_thickness
        if start is None:
            start = int(x_thickness / 2)
        if end is None:
            end = annotation.micron_size['x'] - x_thickness / 2
        self.x_min = start
        self.x_max = self.x_min
        self.x_list = []

        # 2. Core content: a dictionary of slice objects
        self.dict = {}
        while self.x_max <= (end):
            self.x_list = self.x_list + [self.x_max]
            self.dict[self.x_max] = slice(self.x_max, self.x_thickness, anchor_step=anchor_step)
            self.x_max = self.x_max + x_thickness
        self.x_max = self.x_max - x_thickness
        assert self.x_min <= self.x_max, "x_min > x_max"
        self.x_list_valid = [i for i in self.x_list if self.dict[i].anchor is not None]

        # 3. the flat map defined here!
        ind_list = []
        x_list = []
        z_list = []
        x_ccf_list = []  # coordinate in the CCF space
        y_ccf_list = []
        z_ccf_list = []
        region_list = []
        anchor_list = []

        for i, cur_x_position in enumerate(self.x_list_valid):
            if self.dict[cur_x_position].shell is None:
                continue
            cur_shell = self.dict[cur_x_position].shell.copy()
            cur_anchor = self.dict[cur_x_position].anchor.copy()

            ind_list = ind_list + cur_shell.index.tolist()
            x_list = x_list + [cur_x_position / space] * len(cur_shell)
            z_list = z_list + cur_shell.x_expand.tolist()
            x_ccf_list = x_ccf_list + [int(cur_x_position / space)] * len(cur_shell)
            y_ccf_list = y_ccf_list + cur_shell.y.tolist()
            z_ccf_list = z_ccf_list + cur_shell.x.tolist()
            region_list = region_list + cur_shell.region.tolist()
            anchor_list = anchor_list + cur_shell.index.isin(cur_anchor.index.tolist()).tolist()

        self.m2d = pd.DataFrame({'x': x_list,
                                 'z_expand': z_list,
                                 'x_ccf': x_ccf_list,
                                 'y_ccf': y_ccf_list,
                                 'z_ccf': z_ccf_list,
                                 'is_anchor':anchor_list,
                                 'region':region_list
                                 },
                                index=ind_list
                                )
        self.get_boundary()
        return

    def whether_boundary(cur_x_slice, ind, ss):
        assert cur_x_slice in ss.x_list_valid, 'cur_x_slice must be included by ss.x_list_valid'
        if ((cur_x_slice == min(ss.x_list_valid)) | (cur_x_slice == max(ss.x_list_valid))):
            return True

        cur_shell = ss.dict[cur_x_slice].shell
        assert ind in cur_shell.index.tolist(), 'ind must be an index of ss.dict[cur_x_slice].shell'
        x_2d = cur_shell.loc[ind, 'x_expand']
        region = cur_shell.loc[ind, 'region']

        left = cur_shell.loc[cur_shell.x_expand < x_2d]
        right = cur_shell.loc[cur_shell.x_expand > x_2d]

        upper_x_slice = ss.x_list_valid[ss.x_list_valid.index(cur_x_slice) - 1]
        upper = ss.dict[upper_x_slice].shell

        lower_x_slice = ss.x_list_valid[ss.x_list_valid.index(cur_x_slice) + 1]
        lower = ss.dict[lower_x_slice].shell

        for tp_shell in [left, right, upper, lower]:
            if len(tp_shell) == 0:
                return True
            tp_distance = metrics.pairwise_distances([[x_2d]], np.array(tp_shell['x_expand']).reshape(-1, 1))
            tp_neighbor = tp_shell.index[np.argmin(tp_distance)]
            if tp_shell.loc[tp_neighbor, 'region'] != region:
                return True
        return False

    def get_boundary(self):
        boundary_list = []
        for i, cur_x_position in enumerate(self.x_list_valid):
            # if self.dict[cur_x_position].anchor is None:
            #     continue
            # cur_anchor = self.dict[cur_x_position].anchor
            # boundary_list = boundary_list + [slice_set.whether_boundary(cur_x_position, i, self) for i in cur_anchor.index.tolist()]
            if self.dict[cur_x_position].shell is None:
                continue
            cur_shell = self.dict[cur_x_position].shell.copy()
            if ((i == 0) | (i == (len(self.x_list_valid) - 1))):
                cur_shell['is_boundary'] = True
            # else:
            #     cur_shell['is_boundary'] = get_boundary_from_shell(cur_shell)
            boundary_list = boundary_list + [slice_set.whether_boundary(cur_x_position, i, self) for i in cur_shell.index.tolist()]
            # boundary_list = boundary_list + cur_shell.loc[cur_shell.index, 'is_boundary'].tolist()
        self.m2d['is_boundary'] = boundary_list
        return
