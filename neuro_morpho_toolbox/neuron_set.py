import numpy as np
import pandas as pd
from .ml_utilities import *
import time
import os
from sklearn import metrics
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage    
from neuro_morpho_toolbox import neuron, soma_features, projection_features, dendrite_features, lm_dendrite_features, lm_axon_features
import neuro_morpho_toolbox as nmt
import random
from random import randrange
import multiprocessing
def load_swc_list(swc_path, zyx=False,):
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
        # # Test:
        # if len(neurons)>=100:
        #     break
    return neurons

class neuron_set:
    def __init__(self, swc_path=None,  zyx=False, lm_features_path = None, skip_projection=False):
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
        if not skip_projection:
            pf.load_data_from_neuron_dict(self.neurons)
        self.features['projection_features'] = pf
        # print("Getting dendrite features...")
        # df = dendrite_features()
        # df.load_data_from_neuron_dict(self.neurons)
        # self.features['dendrite_features'] = df
        print("Getting metadata...")
        self.metadata['SomaRegion'] = self.features['soma_features'].region.loc[self.names, 'Region']
        hemi_dict = {1:'Left', 2:'Right'}
        self.metadata['Hemisphere'] = [hemi_dict[i] for i in self.features['soma_features'].region.loc[self.names, 'Hemisphere'].tolist()]
        self.metadata['CellType'] = self.features['soma_features'].region.loc[self.names, 'Region'] # Initialized as SomaRegion
        self.metadata['Cluster'] = [0]*len(self.metadata)

        # Zuohan, please complete this: if lm_features_path is not None: load lm features...
        return
    
    def ReduceDimPCA(self, feature_set='projection_features'):
        assert feature_set in self.features.keys(), "Invalid feature_set name."
        if feature_set=='projection_features':
            df = self.features[feature_set].scaled_data
        else:
            df = self.features[feature_set].raw_data
        self.PCA = nmt.PCA_wrapper(df)
        return self.PCA

    def ReduceDimUMAP(self, feature_set='projection_features',
                      n_neighbors=3, min_dist=0.1, n_components=2, metric='euclidean',PCA_first=True,n_PC=100 ):
        # TODO: more reasonable n_PC choice
                      
        assert feature_set in self.features.keys(), "Invalid feature_set name."
        if feature_set=='projection_features':
            df = self.features[feature_set].scaled_data
        else:
            df = self.features[feature_set].raw_data
        self.UMAP = nmt.UMAP_wrapper(df, n_neighbors=n_neighbors,min_dist=min_dist,n_components=n_components,metric=metric,PCA_first=PCA_first,n_PC=n_PC)
        return self.UMAP

    def get_clusters(self,
                    method='SNN_community',
                    karg_dict={'knn':5,
                                'metric':'minkowski',
                                'method':'FastGreedy'},neuron_list = []):
                    #hier_dict={'L_method':'single',
                        #'L_metric':'euclidean',
                        #'criterionH':'inconsistent',
                        # 'depth':2,'R':None,
                        # 't':0.9,
                        # 'optimal_ordering':False},
                    #kmeans_dict={'n_clusters':20, 
                        #         'init':'k-means++', 
                        #        'n_init':10, 'max_iter':300, 'tol':0.0001,
                        #        'precompute_distances':'auto', 
                        #        'verbose':0, 'random_state':None, 
                        #        'copy_x': True, 
                        #       'n_jobs':12, 'algorithm':'auto'}
                    #dbscan_dict={'eps':0.5, 
                        #         'min_samples':5, 
                        #        'metric':'euclidean','metric_params':None,
                        #        'algorithm':'auto', 
                        #        'leaf_size':30, 'p':None,'n_jobs':None}
                    #hdbscan_dict={'min_cluster_size':5, 
                        #         'min_samples':1, 
                        #        'metric':'euclidean','alpha':None,
                        #        'p': ,'algorithm':'auto', 
                        #        'leaf_size':40, 'p':2}
        if len(neuron_list) == 0:
            neuron_list = self.metadata.index
        methods_allowed = ['snn_community', 'hierarchy', 'kmeans', 'dbscan', 'hdbscan']
        assert method.lower() in methods_allowed, "Please set 'method' as one of the following: 'SNN_community', 'Hierarchy', 'Kmeans', 'DBSCAN', 'HDBSCAN'"

        if method.lower()=='snn_community':
            print('Result of SNN_community Clustering')
            if 'knn' in karg_dict.keys():
                knn = karg_dict['knn']
            else:
                knn = 5
            if 'metric' in karg_dict.keys():
                metric = karg_dict['metric']
            else:
                metric = 'minkowski'
            if 'method' in karg_dict.keys():
                community_method = karg_dict['method']
            else:
                community_method = 'FastGreedy'
            cur_clusters = nmt.get_clusters_SNN_community(self.UMAP.loc[neuron_list,:], 
                                knn=knn, metric=metric,method=community_method)
            self.metadata.loc[neuron_list,'Cluster'] = ['C' + str(i) for i in cur_clusters]

        #karg_dict={'L_method':'single','L_metric':'euclidean'.'t':0.9,'criterionH':'inconsistent', depth=2, R=None, monocrit=None}
        if method.lower() =='hierarchy':
            print('Result of Hierarchy Clustering')
            cur_clusters = nmt.get_clusters_Hierarchy_clustering(self.UMAP.loc[neuron_list,:], karg_dict)
            self.metadata.loc[neuron_list,'Cluster'] = ['C' + str(i) for i in cur_clusters]        


        if method.lower() =='kmeans':
            print('Result of Kmeans Clustering')
            cur_clusters = nmt.get_clusters_kmeans_clustering(self.UMAP.loc[neuron_list,:], karg_dict)
            self.metadata.loc[neuron_list,'Cluster'] = ['C' + str(i) for i in cur_clusters]    

        if method.upper() =='DBSCAN':
            print('Result of DBSCAN Clustering')
            cur_clusters = nmt.get_clusters_dbscan_clustering(self.UMAP.loc[neuron_list,:], karg_dict)
            self.metadata.loc[neuron_list,'Cluster'] = ['C' + str(i) for i in cur_clusters]             

        if method.upper() =='HDBSCAN':
            print('Result of HDBSCAN Clustering')
            cur_clusters = nmt.get_clusters_hdbscan_clustering(self.UMAP.loc[neuron_list,:], karg_dict)
            self.metadata.loc[neuron_list,'Cluster'] = ['C' + str(i) for i in cur_clusters]                         
        #self.get_cluster_metric()
        return            
                
        # TODO: other clustering methods...
    def get_cluster_metric(self):
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(self.metadata['CellType'],self.metadata['Cluster']))
        print("Completeness: %0.3f" % metrics.completeness_score(self.metadata['CellType'],self.metadata['Cluster']))
        print("V-measure: %0.3f" % metrics.v_measure_score(self.metadata['CellType'],self.metadata['Cluster']))
        print("Adjusted Rand Index: %0.3f"
        % metrics.adjusted_rand_score(self.metadata['CellType'],self.metadata['Cluster']))
        print("Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(self.metadata['CellType'],self.metadata['Cluster']))
        typeR, typeC = np.unique(self.metadata['Cluster'], return_counts = True)
        if len(typeR)<2:
            print('Number of labels is 1, no available Silhouette Coefficient can be calculated')
        elif len(typeR)>=self.UMAP.shape[0]:
            print('Number of labels is equal to the number of samples, no available Silhouette Coefficient can be calculated')
        else:
            print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(self.UMAP, self.metadata['Cluster'], metric='sqeuclidean'))        

    def get_feature_values(self, feature_name):
        # Search in feature tables
        for cur_featureset_name, cur_featureset in self.features.items():
            if feature_name in cur_featureset.raw_data.columns.tolist():
                if cur_featureset_name == 'projection_features':
                    return pd.DataFrame(cur_featureset.scaled_data[feature_name])
                else:
                    return pd.DataFrame(cur_featureset.raw_data[feature_name])
        # Search in the metadata table
        if feature_name in self.metadata.columns.tolist():
            return pd.DataFrame(self.metadata[feature_name])
        # If cannot find given feature name, return empty dataframe.
        # assert True, "neuron_set.get_feature_values(self, feature_name): Invalid feature_name."
        return pd.DataFrame(index=self.names)

    def get_feature_list_values(self, feature_list):
        assert type(feature_list) == list, "neuron_set.get_feature_list_values: Input must be a list"
        res = pd.DataFrame(index=self.names)
        for feature_name in feature_list:
            tp = self.get_feature_values(feature_name)
            if tp.shape[1]>0:
                res = pd.concat([res, tp], axis=1, sort=False)
        return res

    def FeatureScatter(self, feature_name, map="UMAP"):
        # Find feature values
        if type(feature_name) == list:
            z = self.get_feature_list_values(feature_list=feature_name)
        else:
            z = self.get_feature_values(feature_name=feature_name)
        # Find reduced dimension data
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
        if z.select_dtypes(include=['float', 'int']).shape[1] == z.shape[1]: # If all values are numeric
            fig = nmt.quantitative_scatter(x, y, z)
        else:
            fig = nmt.qualitative_scatter(x, y, z)
        return fig
    
    def load_lm_features_from_folder(self, folder_path):
        '''
        load L-measure features from all .feature files in a folder(currently supporting dendrite, axon and proximal_axon)
        '''
        self.features['lm_dendrite_features'] = lm_dendrite_features()
        self.features['lm_dendrite_features'].load_from_folder(folder_path)
        self.features['lm_dendrite_features'].rearrange_by_id(self.names)
        self.features['lm_axon_features'] = lm_axon_features()
        self.features['lm_axon_features'].load_from_folder(folder_path)
        self.features['lm_axon_features'].rearrange_by_id(self.names)
        return


    def bestCoCluster(self,coclusterDF,axis_color, t = 20, selected_list= None,plotF = False):
        '''
        :param coclusterDF: DataFrane if co-clustering result
        :param axis_color: color for each sample
        :param t: maximum number of cluster
        :param selected_list: list indicating reliable neuron index
        :return: a DataFrame with columns ['ID','Cluster']

        ''' 
        if selected_list == None:
                selected_list =coclusterDF.index.tolist()
        linkmethod = ['single', 'complete','average','weighted','centroid','median','ward']
        paraDF = pd.DataFrame(columns =['method','CCC'],index = linkmethod)
        paraDF.loc[:,'method'] = linkmethod
        for iter_m in linkmethod:
            Y = distance.pdist(np.asarray(coclusterDF))
            Z = linkage(Y, method = iter_m)
            c, coph_dists = hierarchy.cophenet(Z,Y)
            paraDF.loc[iter_m,'cophentic_correlation_dis'] = c
        paraDF.sort_values(by='cophentic_correlation_dis', ascending = False, inplace = True)
        # choose the linkage method which maximizes the cophentic correlation distance
        if type(axis_color) == dict:
            colorDF = pd.DataFrame(index = self.metadata.index,data = self.metadata['CellType'], columns = ['CellType'])
            for iter_idx in colorDF.index:
                colorDF.loc[iter_idx,'Color'] = axis_color[colorDF.loc[iter_idx,'CellType']]
            axis_color = colorDF['Color']
        row_linkage = hierarchy.linkage(distance.pdist(np.asarray(coclusterDF)), method = paraDF.iloc[0,0])
        col_linkage = hierarchy.linkage(distance.pdist(np.asarray(coclusterDF).T), method = paraDF.iloc[0,0])
        cur_clusters = fcluster(row_linkage ,t,criterion='maxclust')
        self.metadata.loc[:,'Cluster'] = ['C' + str(i) for i in cur_clusters]               
        tempARI = metrics.adjusted_rand_score(self.metadata.loc[selected_list,'CellType'],
                                                                                self.metadata.loc[selected_list,'Cluster'])

        print(tempARI)
        if plotF:
            sns.clustermap(coclusterDF, row_linkage = row_linkage, col_linkage = col_linkage, row_colors=axis_color,
                        col_colors = axis_color)#, figsize=(13, 13))#, cmap=sns.diverging_palette(h_neg=150, h_pos=275, s=80, l=55, as_cmap=True))    
        return tempARI


    def pickCLUSTERpara(self, method,selected_list= None):
        '''
        :param method: a str indicating the cluster method
        :param selected_list: list indicating reliable neuron index
        :return: a DataFrame with columns ['ARI', 'NumCluster', 'parameter']
            * For Hierarchy, will try 44986 parameters
            * For Kmeans, will try 60480 parameters
            * For DBSCAN, will try 21200 parameters
            * For HDBSCAN, will be 9198 parameters
            * For SNN_Community, will try 270 parameters
        ''' 
        if selected_list == None:
            selected_list = self.UMAP.index.tolist()
            print('Will calculate ARI for '+ str(len(selected_list) ) + ' neurons')
        result_DF = pd.DataFrame()
        method_list = ['kmeans','snn','hdbscan','hierarchy','dbscan']
        assert method in method_list, "Should be one of "+str(method_list)
        colname = ['ARI','NumCluster','parameter']
        if method.lower() == 'hierarchy':
            #%% Store the result of Hierarchy
            result_hier = pd.DataFrame(columns = colname)
            L_method_list=['single', 'complete','average','weighted','centroid','median','ward']
            L_metric_list=['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
                        'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 
                        'mahalanobis', 'matching','minkowski','rogerstanimoto', 'russellrao',
                        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean']

            criterionH_list=['inconsistent','distance','maxclust','monocrit','maxclust_monocrit']

            hier_dict={'L_method':'single', 'L_metric':'euclidean','criterionH':'inconsistent', 'depth':2,'R':None,
                    't':0.9,'optimal_ordering':False,'colR':3}
            for L_methodidx in L_method_list:
                hier_dict.update(L_method = L_methodidx)
                for L_metricidx in L_metric_list:
                    hier_dict.update(L_metric = L_metricidx )
                    # so far the parameter to generate the linkage array is set
                    if L_methodidx == 'centroid' or L_methodidx == 'median' or L_methodidx == 'ward':
                        if L_metricidx != 'euclidean':
                            continue         
                    for criterionidx in criterionH_list:
                        hier_dict.update(criterionH = criterionidx )   
                        if criterionidx == 'inconsistent' or criterionidx == 'distance':
                            for t_iter in  np.arange(0,1.6,0.05)  : 
                                hier_dict.update(t = t_iter) 
                                if criterionidx == 'inconsistent':
                                    for depth_iter in range(2,16):
                                        hier_dict.update(depth = depth_iter) 
                                        _ = self. get_clusters(method='Hierarchy',karg_dict=hier_dict)
                                        if len(selected_list)==0:
                                            selected_list = self.metadata.index.tolist()
                                        tempARI = metrics.adjusted_rand_score(self.metadata.loc[selected_list,'CellType'],
                                                                            self.metadata.loc[selected_list,'Cluster'])
                                        tempDF = pd.DataFrame([tempARI, 
                                                            len(list(self.metadata.groupby('Cluster'))),
                                                            str(hier_dict)]).T.copy()
                                        tempDF.columns=colname
                                        print(str(hier_dict))
                                        result_hier = result_hier.append(tempDF) 
                                elif criterionidx == 'distance':
                                    _ = self. get_clusters(method='Hierarchy',karg_dict=hier_dict)
                                    tempARI = metrics.adjusted_rand_score(self.metadata.loc[selected_list,'CellType'],
                                                                            self.metadata.loc[selected_list,'Cluster'])
                                    tempDF = pd.DataFrame([tempARI, 
                                                        len(list(self.metadata.groupby('Cluster'))),
                                                        str(hier_dict)]).T.copy()
                                    tempDF.columns=colname
                                    print(str(hier_dict))
                                    result_hier = result_hier.append(tempDF) 
                        if criterionidx == 'maxclust' or criterionidx == 'maxclust_monocrit':
                            for t_iter in  range(20,51): 
                                hier_dict.update(t = t_iter) 
                                _ = self. get_clusters(method='Hierarchy',karg_dict=hier_dict)
                                tempARI = metrics.adjusted_rand_score(self.metadata.loc[selected_list,'CellType'],
                                                                            self.metadata.loc[selected_list,'Cluster'])
                                tempDF = pd.DataFrame([tempARI, 
                                                    len(list(self.metadata.groupby('Cluster'))),
                                                    str(hier_dict)]).T.copy()
                                tempDF.columns=colname
                                print(str(hier_dict))
                                result_hier = result_hier.append(tempDF) 
                                
            idx_hier = ['Hier'+str(x) for x in range(result_hier.shape[0])]    
            result_hier['idx'] = idx_hier
            result_hier.set_index('idx',inplace=True)  
            result_DF = result_hier.copy()
            
        if method.lower() == 'kmeans':
            result_kmeans = pd.DataFrame(columns = colname)
            init_list=['k-means++','random']
            algorithm_list = ['auto','full','elkan']
            precompute_distances_list = ['auto', True, False]
            n_init_list=['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
                        'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 
                        'mahalanobis', 'matching','minkowski','rogerstanimoto', 'russellrao',
                        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean']
            criterionH_list='inconsistent','distance','maxclust','monocrit','maxclust_monocrit'
            kmeans_dict={'n_clusters':20, 'init':'k-means++', 'n_init':10, 'max_iter':300, 'tol':0.0001,
                        'precompute_distances':'auto', 'verbose':0, 'random_state':None,'copy_x': True,
                        'n_jobs':None, 'algorithm':'auto'}
            for init_idx in init_list:
                kmeans_dict.update(init = init_idx)
                for algorithm_idx in algorithm_list:
                    kmeans_dict.update(algorithm = algorithm_idx )
                    for precompute_distances_idx in precompute_distances_list:
                        kmeans_dict.update(precompute_distances = precompute_distances_idx )
                        for n_clustersidx in range(3,45):
                            kmeans_dict.update(n_clusters = n_clustersidx)     
                            for n_initidx in range(7,15):
                                kmeans_dict.update(n_init = n_initidx) 
                                for tol_idx in np.exp(-np.arange(2,4,0.2)):
                                    kmeans_dict.update(tol = tol_idx) 
                                    print(kmeans_dict)
                                    _ = self. get_clusters(method='Kmeans',karg_dict=kmeans_dict)
                                    tempARI = metrics.adjusted_rand_score(self.metadata.loc[selected_list,'CellType'],
                                                                            self.metadata.loc[selected_list,'Cluster'])
                                    tempDF = pd.DataFrame([tempARI, 
                                                        len(list(self.metadata.groupby('Cluster'))),
                                                        str(kmeans_dict)]).T.copy()
                                    tempDF.columns=colname
                                    print(str(kmeans_dict))
                                    result_kmeans = result_kmeans.append(tempDF)         
            idx_kmeans = ['KMeans'+str(x) for x in range(result_kmeans.shape[0])]    
            result_kmeans['idx'] = idx_kmeans
            result_kmeans.set_index('idx',inplace=True)       
            result_DF = result_kmeans.copy()
            
        if method.lower() == 'dbscan':
            result_dbscan = pd.DataFrame(columns = colname)
            algorithm_list = ['auto','ball_tree', 'kd_tree', 'brute']# 
            #
            metriclist = [ 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation','euclidean', 'cosine',
                        'dice','hamming', 'jaccard', 'kulsinski', 'matching','minkowski','rogerstanimoto','russellrao','sokalmichener', 'sokalsneath']
            dbscan_dict={'eps':20, 'min_samples':5, 'metric':'euclidean','metric_params':None, 'algorithm':'auto', 
                        'leaf_size':30, 'p':None,'n_jobs':None}
            for algorithm_idx in algorithm_list:
                dbscan_dict.update(algorithm = algorithm_idx )
                for metric_iter in metriclist:
                    dbscan_dict.update(metric= metric_iter)
                    if algorithm_idx == 'ball_tree' and metric_iter in ['correlation','cosine','sqeuclidean']:
                        continue
                    if algorithm_idx == 'kd_tree' and metric_iter not in ['chebyshev', 'cityblock', 'euclidean',
                                                                                    'infinity', 'l1', 'l2', 'manhattan',
                                                                                    'minkowski', 'p']:
                        continue
                    if algorithm_idx == 'brute' and metric_iter in ['haversine','wminkowski', 'mahalanobis','infinity']:
                        continue
                    if metric_iter in ['wminkowski', 'minkowski']:
                        p_iter = randrange(1,10)
                        dbscan_dict.update(p = p_iter)
                        while metric_iter == 'minkowski' and p_iter == 1:
                            p_iter = randrange(2,10)
                            dbscan_dict.update(p = p_iter)
                    for epsidx in np.exp(-np.arange(0,4,0.5)):
                        dbscan_dict.update(eps = epsidx)
                        for min_samples_iter in range(5,10):
                            dbscan_dict.update(min_samples = min_samples_iter)
                            for leaf_size_iter in range(25,35):
                                dbscan_dict.update(leaf_size = leaf_size_iter)
                                _ = self. get_clusters(method='DBSCAN',karg_dict=dbscan_dict)
                                tempARI = metrics.adjusted_rand_score(self.metadata.loc[selected_list,'CellType'],
                                                                            self.metadata.loc[selected_list,'Cluster'])
                                tempDF = pd.DataFrame([tempARI, len(list(self.metadata.groupby('Cluster'))),str(dbscan_dict)]).T.copy()
                                tempDF.columns=colname
                                print(str(dbscan_dict))
                                result_dbscan = result_dbscan.append(tempDF)        
            idx_dbscan = ['DBSCAN'+str(x) for x in range(result_dbscan.shape[0])]    
            result_dbscan['idx'] = idx_dbscan
            result_dbscan.set_index('idx',inplace=True)     
            result_DF = result_dbscan.copy()
            
        if method.lower() == 'hdbscan':
            result_hdbscan = pd.DataFrame(columns = colname)
            #
            metric_list = [ 'euclidean','minkowski', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra',
                        'chebyshev','correlation','dice', 'hamming', 'jaccard','kulsinski', 'matching', 
                        'rogerstanimoto', 'russellrao','sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
            
        
            algorithm_list = ['best', 'generic','prims_kdtree','boruvka_kdtree']#, 
            cluster_selection_method_list = ['leaf','eom']
            hdbscan_dict={'min_cluster_size':5, 'metric':'euclidean','alpha':1.0, 'min_samples':1,
                        'p':2,'algorithm':'best', 'leaf_size':40, 'approx_min_span_tree':True,
                        'gen_min_span_tree':False,'core_dist_n_jobs':None,'cluster_selection_method':'eom',
                        'allow_single_cluster': False,'prediction_data':False,
                        'match_reference_implementation':False}

            for algorithm_idx in algorithm_list:
                hdbscan_dict.update(algorithm = algorithm_idx)
                for metric_idx in metric_list:
                    if algorithm_idx=='boruvka_kdtree' and metric_idx in['braycurtis','canberra','dice','hamming',
                                                                        'jaccard','kulsinski','matching','rogerstanimoto',
                                                                        'russellrao','sokalmichener', 'sokalsneath']:
                        continue
                    hdbscan_dict.update(metric = metric_idx)
                    for cluster_selection_method_idx in cluster_selection_method_list:
                        hdbscan_dict.update(cluster_selection_method = cluster_selection_method_idx )
                        for alpha_idx in np.arange(0.8,1.5,0.1):
                            hdbscan_dict.update(alpha = alpha_idx)
                            for min_samples_iter in range(1,10):
                                hdbscan_dict.update(min_samples = min_samples_iter)
                                #print(hdbscan_dict)
                                _ = self. get_clusters(method='HDBSCAN',karg_dict=hdbscan_dict)
                                tempARI = metrics.adjusted_rand_score(self.metadata.loc[selected_list,'CellType'],
                                                                            self.metadata.loc[selected_list,'Cluster'])
                                tempDF = pd.DataFrame([tempARI, len(list(self.metadata.groupby('Cluster'))),str(hdbscan_dict)]).T.copy()
                                tempDF.columns = colname
                                print(str(hdbscan_dict))
                                result_hdbscan = result_hdbscan.append(tempDF)     

            idx_hdbscan = ['HDBSCAN'+str(x) for x in range(result_hdbscan.shape[0])]    
            result_hdbscan['idx'] = idx_hdbscan
            result_hdbscan.set_index('idx',inplace=True)       
            result_DF = result_hdbscan.copy()
        if method.lower() == 'snn':
            metric_list = ['sqeuclidean','euclidean','minkowski', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis',
                        'canberra','chebyshev']
            snn_dict = {'knn':5, 'metric':'minkowski','method':'FastGreedy'}
            result_snn= pd.DataFrame(columns = colname)
            for knn_iter in range(3,30):
                snn_dict.update(knn =knn_iter)
                for metric_idx in metric_list:
                    snn_dict.update(metric = metric_idx)
                    _ = self. get_clusters(method='SNN_community',karg_dict=snn_dict)
                    tempARI = metrics.adjusted_rand_score(self.metadata.loc[selected_list,'CellType'],
                                                                            self.metadata.loc[selected_list,'Cluster'])
                    tempDF = pd.DataFrame([tempARI, len(list(self.metadata.groupby('Cluster'))),str(snn_dict)]).T.copy()
                    tempDF.columns=colname
                    print(str(snn_dict))
                    result_snn = result_snn.append(tempDF)
            idx_snn = ['SNN'+str(x) for x in range(result_snn.shape[0])]    
            result_snn['idx'] = idx_snn
            result_snn.set_index('idx',inplace=True)  
            result_DF = result_snn.copy()
        return result_DF.copy()

    def fre_Matrix(self, fre_M, cluster_method, para_input):
        '''
        :param fre_M: a square DataFrame with same row and col name. Here index is same with self.UMAP
        :param cluster_method: a str indicating the cluster method
        :param param para_input: a dataframe at least with column 'parameter' or a dictionary
        :return: a square array with each element being 0 or 1
        '''
        if type(para_input) == dict:
            para_chosen = para_input
        elif type(para_input) == pd.DataFrame:
            para_chosen = eval(para_input.loc[para_input.index.tolist()[randrange(para_input.shape[0])], 'parameter'])
        else:
            print(
                'Input parameters for coclustering must be either dictionary of pandas DataFrame containing all parameters')
        clusterL = self.metadata.index[
            random.sample(range(0, self.metadata.shape[0]), int(self.metadata.shape[0] * 0.95))]
        _ = self.get_clusters(method=cluster_method, karg_dict=para_chosen, neuron_list=clusterL)
        Crange, Ccounts = np.unique(self.metadata.loc[clusterL, 'Cluster'], return_counts=True)
        for iter_C in Crange:
            selected_row = self.metadata.loc[clusterL, :]
            selected_row = selected_row[selected_row["Cluster"] == iter_C]
            Clist = selected_row.index.tolist()
            fre_M.loc[Clist, Clist] = fre_M.loc[Clist, Clist] + 1
        return fre_M.values

    def para_cocluster(self, cluster_method, corenum, run_num, para_input):
        '''
        :param cluster_method: a str indicating the cluster method
        :param para_input: a dataframe at least with column 'parameter' or a dictionary
        :param corenum: an int indicating number of cores to use
        :param run_num: number of co-clustering
        :return: a square array with each element indicating number of cocluster within run_num
        '''
        assert type(para_input) == dict or type(
            para_input) == pd.DataFrame, "Input parameters for coclustering must be either dictionary of pandas DataFrame containing all parameters"
        start = time.perf_counter()
        start = time.time()
        cores = corenum  # multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        fre_M_t = pd.DataFrame(index=self.UMAP.index, columns=self.UMAP.index)
        fre_M_t[fre_M_t.isnull()] = 0
        pool_list = []
        result_list = []
        for i in range(run_num):
            pool_list.append(pool.apply_async(self.fre_Matrix, (fre_M_t, cluster_method, para_input)))
        result_list = [xx.get() for xx in pool_list]
        print(sum([xx for xx in result_list]))
        pool.close()
        pool.join()
        elapsed = (time.time() - start)
        print('Time needed to run Hierarchy is ' + str(elapsed))
        return sum([xx for xx in result_list])