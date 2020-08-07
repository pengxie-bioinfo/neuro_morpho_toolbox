# import numpy as np
# import pandas as pd
# from sklearn import metrics
# from sklearn.metrics.pairwise import pairwise_distances
#
# from .utilities import cart2pol_3d
# from .brain_structure import *
# from neuro_morpho_toolbox import annotation,bs,neuron,ss
#
# ##################################################################
# # Global variables
# ##################################################################
# CTX_regions = [bs.id_to_name(i) for i in bs.get_all_child_id('CTX') if i in bs.selected_regions]
# CTX_regions = sorted(CTX_regions)
#
# def get_segments(swc):
#     # lab = [i for i,name in enumerate(self.swc.index.tolist()) if self.swc.loc[name, "parent"]!=(-1)]
#     # child = self.swc[self.swc.parent != (-1)]
#     child = swc[swc.parent.isin(swc.index)]
#     parent = swc.loc[child.parent]
#     rho, theta, phi = cart2pol_3d(np.array(child[["x", "y", "z"]]) - np.array(parent[["x", "y", "z"]]))
#     res = pd.DataFrame({"type": child.type,
#                         "rho": rho,
#                         "theta": theta,
#                         "phi": phi,
#                         "x": (np.array(child.x) + np.array(parent.x)) / 2,
#                         "y": (np.array(child.y) + np.array(parent.y)) / 2,
#                         "z": (np.array(child.z) + np.array(parent.z)) / 2
#                         })
#     res.index = range(1, len(child) + 1)
#     # soma
#     soma = swc[((swc.type == 1) & (swc.parent == -1))]
#     if len(soma) > 0:
#         soma_res = pd.DataFrame({"type": 1,
#                                  "rho": 1,
#                                  "theta": 0,
#                                  "phi": 0,
#                                  "x": soma.x.iloc[0],
#                                  "y": soma.y.iloc[0],
#                                  "z": soma.z.iloc[0]},
#                                 index=[0])
#         res = soma_res.append(res)
#     return res
#
# def get_point_set(df, projection_feild, x_position, x_thickness=1000):
#     ps = df.loc[((df.x>(x_position-x_thickness/2)) & (df.x<=(x_position+x_thickness/2))), ['z', 'y', 'rho']]
#     if len(ps)==0:
#         return None
#     ps = ps.rename(columns={'z':'x'})
#     ps.x = (ps.x / space).astype('int32')
#     ps.y = (ps.y / space).astype('int32')
#     ps['voxel'] = [str(int(ps.loc[i, 'x']))+"_"+str(int(ps.loc[i, 'y'])) for i in ps.index.tolist()]
#     # Only consider points in the cortex
#     ps = ps[ps.voxel.isin(projection_feild.index.tolist())]
#     if len(ps)==0:
#         return None
#     ps['anchor'] = projection_feild.loc[ps.voxel.tolist(), 'anchor'].tolist()
#     return ps
#
#
# '''
# Define 'arbor' class, which is a cluster of branches.
# '''
# class arbor:
#     def __init__(self, nn, arbor_id):
#         self.name = nn.name + "_" + str(arbor_id)
#         self.swc = nn.swc[nn.swc.arbor_id == arbor_id].copy()
#         self.nodes = self.swc.index.tolist()
#         self.valid = True
#         self.get_total_length()
#         self.get_link_to_trunk(nn)
#         self.get_region()
#         self.is_ctx = (self.region in CTX_regions)
#
#         ##################################################################################
#         # 1. Segments & center
#         ##################################################################################
#         self.seg = get_segments(self.swc)
#
#         center_name, self.outlier_ratio = self.get_cluster_center(silent=True)  # outlier points will be removed
#         self.center = self.seg.loc[center_name]
#
#         ##################################################################################
#         # 2. For cortical arbors
#         ##################################################################################
#
#         self.get_surface(ss)
#         if len(self.surface) == 0:
#             self.valid = False
#             return
#         else:
#             distance = metrics.euclidean_distances(self.surface[['2d_x', '2d_y']])
#             cid = np.argmin(np.mean(distance, axis=1))
#         self.surface_center = self.surface.iloc[cid, :]
#         #         self.surface_area = len(self.surface.anchor.value_counts())  # TODO: consider anchor area
#         self.get_depth(ss)
#
#         return
#
#     def get_total_length(self):
#         # To be implemented
#         self.total_length = 9999
#         return
#
#     def whether_single_tree(self):
#         # To be implemented
#         self.is_single_tree = True
#         return
#
#     def get_link_to_trunk(self, nn):
#         assert 'is_trunk' in self.swc.columns.tolist(), 'no trunk information found in neuron tree'
#         tp = self.swc[self.swc['is_trunk']==False].index.tolist()
#         tp = [i for i in tp if nn.swc.loc[i, 'parent'] in nn.swc.index.tolist()]
#         tp = nn.swc.loc[nn.swc.loc[tp, 'parent'], ['is_trunk']].copy()
#         tp = tp[tp['is_trunk']]
#         self.connected_to_trunk = False
#         self.link = None
#         if len(tp)>0:
#             self.connected_to_trunk = True
#             tp = tp.index.tolist()[0]
#             self.link = {'trunk_node' : nn.swc.loc[tp, 'parent'],
#                          'arbor_node' : tp
#                          }
#         # To be implemented
#         # self.order = 0
#         return
#
#     def get_cluster_center(self, silent=True):
#         df = self.swc.copy()[['x', 'y', 'z']]
#         distance = metrics.euclidean_distances(df)
#         cid = np.argmin(np.mean(distance, axis=1))
#         cid = df.index.tolist()[cid]
#         # exclude outliers
#         md = metrics.euclidean_distances(np.array(df.loc[cid]).reshape(1, 3), df).reshape(-1, )
#         #         tp = df[md>(np.median(md)+2*min(500, np.std(md)))] # define outliers
#         tp = df[md > 1000]  # define outliers
#         outlier_ratio = len(tp) / len(df) * 100
#         if not silent:
#             if outlier_ratio > 10:
#                 print("%s: %.2f%% arbor excluded." % (self.name, outlier_ratio))
#         self.swc.drop(index=tp.index, inplace=True)
#         # Re-center
#         df = self.swc.copy()[['x', 'y', 'z']]
#         distance = metrics.euclidean_distances(df)
#         cid = np.argmin(np.mean(distance, axis=1))
#         cid = df.index.tolist()[cid]
#         self.center = self.swc.loc[[cid]]
#         return
#
#     def get_region(self):
#         self.region = None
#         arbor_int = self.center.copy()
#         arbor_int['x'] = arbor_int['x'] / annotation.space['x']
#         arbor_int['y'] = arbor_int['y'] / annotation.space['y']
#         arbor_int['z'] = arbor_int['z'] / annotation.space['z']
#         arbor_int = arbor_int.round(0).astype(int)
#         if ((arbor_int.x.iloc[0] >= 0) & (arbor_int.x.iloc[0] < annotation.size['x']) &
#                 (arbor_int.y.iloc[0] >= 0) & (arbor_int.y.iloc[0] < annotation.size['y']) &
#                 (arbor_int.z.iloc[0] >= 0) & (arbor_int.z.iloc[0] < annotation.size['z'])
#         ):
#             arbor_region_id = annotation.array[arbor_int.x.iloc[0],
#                                               arbor_int.y.iloc[0],
#                                               arbor_int.z.iloc[0]
#             ]
#             if arbor_region_id in list(bs.dict_to_selected.keys()):
#                 arbor_region_id = bs.dict_to_selected[arbor_region_id]
#                 self.region = bs.id_to_name(arbor_region_id)
#             else:
#                 print('%s:\t%s' % (self.name, arbor_region_id))
#         return
#
#     ##########################################################################################
#     # Functions for flat map
#     ##########################################################################################
#     def get_surface(self, ss):
#         ps = pd.DataFrame(columns=['x', 'y', 'rho', 'voxel', 'anchor'])
#         for cur_x_position in ss.x_list_valid:
#             cur_shell = ss.dict[cur_x_position].shell
#             cur_pf = ss.dict[cur_x_position].pf
#             if cur_shell is None:
#                 continue
#             cur_ps = get_point_set(self.seg,
#                                    projection_feild=cur_pf,
#                                    x_position=cur_x_position,
#                                    x_thickness=ss.x_thickness
#                                    )
#             if cur_ps is None:
#                 continue
#             ps = pd.concat([ps, cur_ps], axis=0)
#         ps = ps[['anchor', 'rho']]
#         ps['2d_x'] = [anchor_df.loc[i, 'z_expand'] for i in ps.anchor.tolist()]
#         ps['2d_y'] = [anchor_df.loc[i, 'x'] for i in ps.anchor.tolist()]
#         ps['2d_area'] = [anchor_df.loc[i, 'area'] for i in ps.anchor.tolist()]
#         self.surface = ps
#
#         # Calculate surface area
#         # sort anchor by sum of rho
#         wdf = pd.DataFrame(ps.groupby('anchor')['rho'].sum())
#         wdf = wdf.sort_values(['rho'], ascending=False)
#         # Find anchors that cover x% of rho
#         wdf = wdf.cumsum() / wdf['rho'].sum() * 100
#         wdf = wdf.iloc[:((wdf['rho'] < 95).sum() + 1), :]
#         # sum up area of each anchor
#         self.surface_area = anchor_df.loc[wdf.index, 'area'].sum()
#         return
#
#     def get_depth(self, ss):
#         depth = pd.DataFrame(columns=['depth', 'rho'])
#         if len(self.surface) == 0:
#             return depth
#         df = pd.concat([self.surface[['anchor']],
#                         self.seg.loc[self.surface.index, ['x', 'y', 'z', 'rho']]
#                         ], axis=1)
#         anchor_list = df.anchor.value_counts().index.tolist()
#         prof = []
#         ind_list = []
#         for cur_anchor in anchor_list:
#             cur_df = np.array(df.loc[df.anchor == cur_anchor, ['z', 'y']]).reshape(-1, 2)
#             if len(cur_df) == 0:
#                 continue
#             cur_df = cur_df - np.array(anchor_df.loc[cur_anchor, ['z_ccf', 'y_ccf']] * space).reshape(1, 2)
#             anchor_vector = my_normalize(np.array(anchor_df.loc[cur_anchor, ['vx', 'vy']]))
#             anchor_vector = anchor_vector.reshape(-1, 1)
#             prof = prof + list(my_projection(cur_df, anchor_vector).reshape(-1, ))
#             ind_list = ind_list + df.index[df.anchor == cur_anchor].tolist()
#         self.depth = pd.DataFrame({'depth': prof,
#                                    'rho': df.rho[ind_list],
#                                    },
#                                   index=ind_list)
#         return
#
#     def plot_arbor(self, ax=None, x_shift=0, y_shift=0):
#         if ax is None:
#             fig, ax = plt.subplots(1, 1, figsize=(2.5, 3))
#             ax.invert_yaxis()
#             ax.set_ylim(1000, -200)
#             ax.set_xlim(-500, 500)
#         ax.plot(self.rot_seg.Xe + x_shift,
#                 self.rot_seg.Ye + y_shift)
#         ax.plot(self.center.rx + x_shift, self.center.ry + y_shift, c='k', marker='o')
#         return ax
#
#
# class arbor_neuron(neuron):
#     def __init__(self, file, trunk_file, zyx=False, registered=True,
#                  scale=None, trunk_scale=None, dist_thres=0.01):
#         '''
#         What's special about the swc file of arbor_neuron?
#         It has an additional column called label, as ID of the arbor
#         '''
#         super(arbor_neuron, self).__init__(file, zyx=zyx, registered=registered, scale=scale)
#         n_skip = 0
#         with open(self.file, "r") as f:
#             for line in f.readlines():
#                 line = line.strip()
#                 if line.startswith("#"):
#                     n_skip += 1
#                 else:
#                     break
#         f.close()
#         tp = pd.read_csv(self.file, index_col=0, skiprows=n_skip, sep=" ",
#                           usecols=[0, 7],
#                           names=['##n', 'arbor_id']
#                           )
#         self.swc['arbor_id'] = tp['arbor_id'].tolist()
#         self.get_degree()
#         if trunk_scale is None:
#             trunk_scale = scale
#         self.get_trunk(neuron(trunk_file, zyx=zyx, registered=registered, scale=trunk_scale),
#                        dist_thres=dist_thres)
#         self.get_arbors()
#         self.get_topology(dist_thres=dist_thres)
#         return
#
#     def get_arbors(self):
#         alist = sorted(self.swc.arbor_id.value_counts().index.tolist())
#         self.arbor_names = []
#         self.arbors = {}
#         self.arbor_df = pd.DataFrame(columns=['region'])
#         for cid in alist:
#             cname = self.name + "_" + str(cid)
#             # Skip arbors with too few branch points
#             cswc = self.swc[self.swc.arbor_id == cid].copy()
#             n_branch = (cswc['degree'] > 2).sum()
#             if n_branch == 0:
#                 continue
#             carbor = arbor(self, cid)
#             # Skip arbors that are too short
#             if ((carbor.valid != True) | (carbor.total_length < 1000)):
#                 continue
#             self.arbors[cname] = carbor
#             self.arbor_names.append(cname)
#             self.arbor_df.loc[cname, 'region'] = carbor.region
#         return
#
#     def get_trunk(self, tr, dist_thres=0.01):
#         dist = pairwise_distances(self.swc[['x', 'y', 'z']], tr.swc[['x', 'y', 'z']])
#         mindist = pd.DataFrame(np.min(dist, axis=1), index=self.swc.index, columns=['dist_to_trunk'])
#         self.swc['is_trunk'] = (mindist['dist_to_trunk']<dist_thres)
#         # To be implemented
#         # save trunk as a separate tree, with index matching 'neuron'
#         mindist['min_id'] = tr.swc.iloc[np.argmin(dist, axis=1)].index.tolist()
#         mindist = mindist[mindist['dist_to_trunk']<dist_thres]
#         rename_dict = dict(zip(mindist['min_id'].tolist(), mindist.index.tolist()))
#         rename_dict[-1] = -1
#         self.trunk_swc = tr.swc.copy()[["type", "x", "y", "z", "r", "parent"]]
#         # self.trunk_swc[["x", "y", "z"]] = self.trunk_swc[["x", "y", "z"]].round(3)
#         self.trunk_swc.rename(index=rename_dict, inplace=True)
#         self.trunk_swc['parent'] = self.trunk_swc['parent'].map(rename_dict)
#         return
#
#     def get_topology(self, save=None, dist_thres=0.01):
#         '''
#         :param dist_thres:
#         :return: an swc of the topology graph
#         '''
#         res = self.swc.copy()
#         res['is_topology'] = False
#         res.loc[res['is_trunk'], 'is_topology'] = True
#         for cname in self.arbor_names:
#             carbor = self.arbors[cname]
#             carbor.get_cluster_center()
#             dist = pairwise_distances(carbor.center[['x', 'y', 'z']], self.swc[['x', 'y', 'z']])
#             cid = self.swc.index[np.argmin(dist)]
#             res.loc[cid, 'type'] = 5
#             res.loc[cid, 'r'] = 10
#
#             ## trace back to trunk
#             while not res.loc[cid, 'is_topology']:
#                 res.loc[cid, 'is_topology'] = True
#                 cid = res.loc[cid, 'parent']
#                 if cid not in res.index.tolist():
#                     break
#         res = res[res['is_topology']]
#         if save is not None:
#             res.iloc[:, :6].to_csv(save, sep=" ")
#         self.topo = res
#         return
#
#
#
