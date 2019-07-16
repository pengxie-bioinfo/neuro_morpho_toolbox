import numpy as np
import pandas as pd

import time
import os

from neuro_morpho_toolbox import neuron, soma_features, projection_features
import neuro_morpho_toolbox as nmt

def load_swc_list(swc_path, zyx=False):
    '''
    load all the swc files under a folder
    :param swc_path:
    :return: dict of neurons
    '''
    neurons = {}
    start = time.time()
    for swc_file in sorted(os.listdir(swc_path)):
        if not swc_file.endswith(("swc", "SWC")):
            continue
        # print(os.path.join(swc_path, swc_file))
        cur_neuron = neuron(os.path.join(swc_path, swc_file))
        if cur_neuron.pass_qc():
            neurons[cur_neuron.name] = cur_neuron
        else:
            print("QC failed: %s" % (swc_file))
        if len(neurons) % 100 == 0:
            print("%d loaded: %.1fs" % (len(neurons), time.time()-start))
            start = time.time()
        # Test:
        if len(neurons)>=1:
            break
    return neurons

class neuron_set:
    def __init__(self, swc_path=None, zyx=False):
        '''
        load all the
        :param path:
        :param zyx:
        '''
        self.names = []
        self.neurons = {}
        self.metadata = pd.DataFrame()
        self.features = {}
        if swc_path is None:
            return
        else:
            print("Loading...")
            self.neurons = load_swc_list(swc_path, zyx)
            self.names = list(self.neurons.keys())
            self.metadata = pd.DataFrame(index=self.names)
            print("Finding soma locations...")
            sf = soma_features()
            sf.load_data_from_neuron_dict(self.neurons)
            self.features['soma_features'] = sf
            print("Getting projection features...")
            pf = projection_features()
            pf.load_data_from_neuron_dict(self.neurons)
            self.features['projection_features'] = pf
        return

    def ReduceDimPCA(self, feature_set='projection_features'):
        assert feature_set in self.features.keys(), "Invalid feature_set name."
        df = self.features[feature_set].raw_data # TODO: take normalized data as input
        self.PCA = nmt.PCA_wrapper(df)
        return self.PCA

    def ReduceDimUMAP(self, feature_set='projection_features',
                      n_neighbors=3, min_dist=0.1, n_components=2, metric='euclidean'):
        assert feature_set in self.features.keys(), "Invalid feature_set name."
        df = self.features[feature_set].raw_data  # TODO: take normalized data as input
        self.UMAP = nmt.UMAP_wrapper(df,
                                     n_neighbors=n_neighbors,
                                     min_dist=min_dist,
                                     n_components=n_components,
                                     metric=metric)
        return self.UMAP

    def get_feature_values(self, feature_name):
        for _, cur_feature in self.features.items():
            if feature_name in cur_feature.raw_data.columns.tolist():
                return cur_feature.raw_data[feature_name]
        if feature_name == "Soma_region":
            return self.features["soma_features"].region
        assert True, "neuron_set.get_feature_values(self, feature_name): Invalid feature_name."
        return

    def FeatureScatter(self, feature_name, map="UMAP"):
        # Find feature values
        z = self.get_feature_values(feature_name=feature_name)
        if map == "UMAP":
            x = self.UMAP.iloc[:,0]
            y = self.UMAP.iloc[:,1]
        elif map == "PCA":
            x = self.PCA.iloc[:, 0]
            y = self.PCA.iloc[:, 1]
        else:
            assert True, "neuron_set.FeatureScatter(self, feature_name, map='UMAP'): Invalid map."
            return
        # If feature is categorical
        if not z.dtype in ['int', 'float']:
            fig = nmt.qualitative_scatter(x, y, z)
        else:
            fig = nmt.quantitative_scatter(x, y, z)
        return fig


