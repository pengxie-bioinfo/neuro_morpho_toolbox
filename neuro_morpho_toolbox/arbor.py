import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances

from .utilities import cart2pol_3d
from .brain_structure import *
from neuro_morpho_toolbox import annotation,bs,neuron

'''
Define 'arbor' class, which is a cluster of branches.
'''
class arbor:
    def __init__(self, nn, arbor_id):
        self.name = nn.name + "_" + str(arbor_id)
        self.swc = nn.swc[nn.swc.arbor_id == arbor_id].copy()
        self.nodes = self.swc.index.tolist()
        self.valid = True
        self.get_total_length()
        self.get_link_to_trunk(nn)
        self.get_region()
        return

    def get_total_length(self):
        # To be implemented
        self.total_length = 9999
        return

    def whether_single_tree(self):
        # To be implemented
        self.is_single_tree = True
        return

    def get_link_to_trunk(self, nn):
        assert 'is_trunk' in self.swc.columns.tolist(), 'no trunk information found in neuron tree'
        tp = self.swc[self.swc['is_trunk']==False].index.tolist()
        tp = [i for i in tp if nn.swc.loc[i, 'parent'] in nn.swc.index.tolist()]
        tp = nn.swc.loc[nn.swc.loc[tp, 'parent'], ['is_trunk']].copy()
        tp = tp[tp['is_trunk']]
        self.connected_to_trunk = False
        self.link = None
        if len(tp)>0:
            self.connected_to_trunk = True
            tp = tp.index.tolist()[0]
            self.link = {'trunk_node' : nn.swc.loc[tp, 'parent'],
                         'arbor_node' : tp
                         }
        # To be implemented
        # self.order = 0
        return

    def get_cluster_center(self, silent=True):
        df = self.swc.copy()[['x', 'y', 'z']]
        distance = metrics.euclidean_distances(df)
        cid = np.argmin(np.mean(distance, axis=1))
        cid = df.index.tolist()[cid]
        # exclude outliers
        md = metrics.euclidean_distances(np.array(df.loc[cid]).reshape(1, 3), df).reshape(-1, )
        #         tp = df[md>(np.median(md)+2*min(500, np.std(md)))] # define outliers
        tp = df[md > 1000]  # define outliers
        outlier_ratio = len(tp) / len(df) * 100
        if not silent:
            if outlier_ratio > 10:
                print("%s: %.2f%% arbor excluded." % (self.name, outlier_ratio))
        self.swc.drop(index=tp.index, inplace=True)
        # Re-center
        df = self.swc.copy()[['x', 'y', 'z']]
        distance = metrics.euclidean_distances(df)
        cid = np.argmin(np.mean(distance, axis=1))
        cid = df.index.tolist()[cid]
        self.center = self.swc.loc[[cid]]
        return

    def get_region(self):
        self.region = None
        arbor_int = self.swc.copy()
        arbor_int['x'] = arbor_int['x'] / annotation.space['x']
        arbor_int['y'] = arbor_int['y'] / annotation.space['y']
        arbor_int['z'] = arbor_int['z'] / annotation.space['z']
        arbor_int = arbor_int.round(0).astype(int)
        if ((arbor_int.x.iloc[0] >= 0) & (arbor_int.x.iloc[0] < annotation.size['x']) &
                (arbor_int.y.iloc[0] >= 0) & (arbor_int.y.iloc[0] < annotation.size['y']) &
                (arbor_int.z.iloc[0] >= 0) & (arbor_int.z.iloc[0] < annotation.size['z'])
        ):
            arbor_region_id = annotation.array[arbor_int.x.iloc[0],
                                              arbor_int.y.iloc[0],
                                              arbor_int.z.iloc[0]
            ]
            if arbor_region_id in list(bs.dict_to_selected.keys()):
                arbor_region_id = bs.dict_to_selected[arbor_region_id]
                self.region = bs.id_to_name(arbor_region_id)
            else:
                print('%s:\t%s' % (self.name, arbor_region_id))
        return


class arbor_neuron(neuron):
    def __init__(self, file, trunk_file, zyx=False, registered=True,
                 scale=None, trunk_scale=None, dist_thres=0.01):
        '''
        What's special about the swc file of arbor_neuron?
        It has an additional column called label, as ID of the arbor
        '''
        super(arbor_neuron, self).__init__(file, zyx=zyx, registered=registered, scale=scale)
        n_skip = 0
        with open(self.file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith("#"):
                    n_skip += 1
                else:
                    break
        f.close()
        tp = pd.read_csv(self.file, index_col=0, skiprows=n_skip, sep=" ",
                          usecols=[0, 7],
                          names=['##n', 'arbor_id']
                          )
        self.swc['arbor_id'] = tp['arbor_id'].tolist()
        self.get_degree()
        if trunk_scale is None:
            trunk_scale = scale
        self.get_trunk(neuron(trunk_file, zyx=zyx, registered=registered, scale=trunk_scale),
                       dist_thres=dist_thres)
        self.get_arbors()
        self.get_topology(dist_thres=dist_thres)
        return

    def get_arbors(self):
        alist = sorted(self.swc.arbor_id.value_counts().index.tolist())
        self.arbor_names = []
        self.arbors = {}
        self.arbor_df = pd.DataFrame(columns=['region'])
        for cid in alist:
            cname = self.name + "_" + str(cid)
            # Skip arbors with too few branch points
            cswc = self.swc[self.swc.arbor_id == cid].copy()
            n_branch = (cswc['degree'] > 2).sum()
            if n_branch == 0:
                continue
            carbor = arbor(self, cid)
            # Skip arbors that are too short
            if ((carbor.valid != True) | (carbor.total_length < 1000)):
                continue
            self.arbors[cname] = carbor
            self.arbor_names.append(cname)
            self.arbor_df.loc[cname, 'region'] = carbor.region
        return

    def get_trunk(self, tr, dist_thres=0.01):
        dist = pairwise_distances(self.swc[['x', 'y', 'z']], tr.swc[['x', 'y', 'z']])
        mindist = pd.DataFrame(np.min(dist, axis=1), index=self.swc.index, columns=['dist_to_trunk'])
        self.swc['is_trunk'] = (mindist['dist_to_trunk']<dist_thres)
        # To be implemented
        # save trunk as a separate tree, with index matching 'neuron'
        mindist['min_id'] = tr.swc.iloc[np.argmin(dist, axis=1)].index.tolist()
        mindist = mindist[mindist['dist_to_trunk']<dist_thres]
        rename_dict = dict(zip(mindist['min_id'].tolist(), mindist.index.tolist()))
        rename_dict[-1] = -1
        self.trunk_swc = tr.swc.copy()[["type", "x", "y", "z", "r", "parent"]]
        # self.trunk_swc[["x", "y", "z"]] = self.trunk_swc[["x", "y", "z"]].round(3)
        self.trunk_swc.rename(index=rename_dict, inplace=True)
        self.trunk_swc['parent'] = self.trunk_swc['parent'].map(rename_dict)
        return

    def get_topology(self, save=None, dist_thres=0.01):
        '''
        :param dist_thres:
        :return: an swc of the topology graph
        '''
        res = self.swc.copy()
        res['is_topology'] = False
        res.loc[res['is_trunk'], 'is_topology'] = True
        for cname in self.arbor_names:
            carbor = self.arbors[cname]
            carbor.get_cluster_center()
            dist = pairwise_distances(carbor.center[['x', 'y', 'z']], self.swc[['x', 'y', 'z']])
            cid = self.swc.index[np.argmin(dist)]
            res.loc[cid, 'type'] = 5
            res.loc[cid, 'r'] = 10

            ## trace back to trunk
            while not res.loc[cid, 'is_topology']:
                res.loc[cid, 'is_topology'] = True
                cid = res.loc[cid, 'parent']
                if cid not in res.index.tolist():
                    break
        res = res[res['is_topology']]
        if save is not None:
            res.iloc[:, :6].to_csv(save, sep=" ")
        self.topo = res
        return



