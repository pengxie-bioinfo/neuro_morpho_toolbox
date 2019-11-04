import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from joblib import Memory
import os
import re
import pickle
from timeit import default_timer as timer
from sklearn.preprocessing import scale
from sklearn.manifold import Isomap, TSNE
from sklearn import metrics
from random import randrange
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import scipy
from sklearn.neighbors import KDTree, BallTree
import umap
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster,inconsistent
from scipy.cluster.hierarchy import maxRstat,maxinconsts
import sklearn.cluster
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu
import hdbscan
# from pysankey import sankey

from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering, Birch
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics
import igraph
from math import ceil
from timeit import default_timer as timer
import subprocess
from pathlib import Path
from numpy import linalg as LA

def helloworld():
    print('Hello World')
#######################################################################################
# Functions for dimension reduction

def PCA_wrapper(df, n_components=50):
    Z = PCA(n_components=n_components).fit_transform(df)
    Z_df = pd.DataFrame(Z, index=df.index)
    return Z_df

def UMAP_wrapper(df, n_neighbors=3, min_dist=0.1, n_components=2, metric='euclidean', PCA_first=True, n_PC=100):
    if PCA_first:
        n_PC = min([df.shape[0], n_PC, df.shape[1]])
        pca = PCA(n_components=n_PC)
        Z = pca.fit_transform(df)
        df = pd.DataFrame(Z, index=df.index)
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors,
                             min_dist=min_dist,
                             n_components=n_components,
                             metric=metric)
    Z = umap_reducer.fit_transform(df)
    Z_df = pd.DataFrame(Z, index=df.index)
    return Z_df

#######################################################################################


# def df_filter(df):
#     ind = []
#     for i in range(df.shape[0]):
#         if all(df.iloc[i, :] != ' -nan'):
#             ind.append(i)
#     df = df.astype('float64')
#     return ([df.iloc[ind, :], ind])


# def feature_hist(df, ncol=4, len_single_plot=5):
#     n = df.shape[1]
#     if ((n % ncol) == 0):
#         nrow = n / ncol
#     else:
#         nrow = int(n / ncol) + 1
#     fig, ax = plt.subplots(nrow, ncol, figsize=(len_single_plot * ncol, len_single_plot * nrow))
#     ax = ax.reshape(-1, )
#     for i in range(n):
#         ax[i].hist(df.iloc[:, i], label=df.columns[i])
#         ax[i].legend()
#     return

#
# def features_scatter(X, df, ncol=4, len_single_plot=3.5):
#     n = df.shape[1]
#     if ((n % ncol) == 0):
#         nrow = n / ncol
#     else:
#         nrow = int(n / ncol) + 1
#     fig, ax = plt.subplots(nrow, ncol, figsize=(len_single_plot * ncol, len_single_plot * nrow))
#     ax = ax.reshape(-1, )
#     for i in range(n):
#         ax[i].scatter(X[:, 0], X[:, 1], c=df.iloc[:, i], cmap='coolwarm')
#         ax[i].set_title(df.columns[i])
#     return


def match1d(query, target):
    query = query.tolist()
    target = target.tolist()
    target = dict(zip(target, range(len(target))))
    if (set(query).issubset(set(target.keys()))):
        result = [target[i] for i in query]
        return (np.array(result))
    else:
        print("Query should be a subset of target!")
        return


def SNN(x, k=3, verbose=True, metric='minkowski'):
    '''
    x: n x m matrix, n is #sample, m is #feature
    '''
    n, m = x.shape
    # Find a ranklist of neighbors for each sample
    timestamp = timer()
    if not verbose:
        print('Create KNN matrix...')
    knn = NearestNeighbors(n_neighbors=n, metric=metric)
    knn.fit(x)
    A = knn.kneighbors_graph(x, mode='distance')
    A = A.toarray()
    A_rank = A
    for i in range(n):
        A_rank[i, :] = np.argsort(A[i, :])
    A_rank = np.array(A_rank, dtype='int')
    A_knn = A_rank[:, :k]
    if not verbose:
        print("Time elapsed:\t", timer() - timestamp)

    # Create weighted edges between samples
    timestamp = timer()
    if not verbose:
        print('Generate edges...')
    edge = []
    for i in range(n):
        for j in range(i + 1, n):
            shared = set(A_knn[i, :]).intersection(set(A_knn[j, :]))
            shared = np.array(list(shared))
            if (len(shared) > 0):  # When i and j have shared knn
                strength = k - (match1d(shared, A_knn[i, :]) + match1d(shared, A_knn[j, :]) + 2) / 2
                strength = max(strength)
                if (strength > 0):
                    edge = edge + [i + 1, j + 1, strength]
    edge = np.array(edge).reshape(-1, 3)
    if not verbose:
        print("Time elapsed:\t", timer() - timestamp)
    return (edge)


def get_clusters_SNN_community(x, knn=3, metric='minkowski', method='FastGreedy'):
    stamp = timer()
    '''
    Create graph
    '''
    edge = SNN(x, knn, metric=metric)
    g = igraph.Graph()
    # in order to add edges, we have to add all the vertices first
    # iterate through edges and put all the vertices in a list
    vertex = []
    edge_list = [(int(edge[i, 0]) - 1, int(edge[i, 1]) - 1) for i in range(len(edge))]
    for e in edge_list:
        vertex.extend(e)
    g.add_vertices(list(set(vertex)))  # add a list of unique vertices to the graph
    g.add_edges(edge_list)  # add the edges to the graph.
    g.es['weights'] = edge[:, 2]
    '''
    Clustering
    '''
    # print('Time elapsed:\t', '{:.2e}'.format(timer() - stamp))
    # TODO: other community detection methods...
    if method == 'FastGreedy':
        return (g.community_fastgreedy(weights='weights').as_clustering().membership)
    elif method == "Louvain":
        return (g.community_multilevel(weights='weights', return_levels=False).membership)
    else:
        return


def get_clusters_Hierarchy_clustering(x, hier_dict):  
    #default value
    L_method='single'
    L_metric='euclidean'  
    t = 0.9
    criterionH='inconsistent'
    depth = 2
    R = None
    colR = 3
    #L_metric can be 'braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, 
                    #‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’,
                    # ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
                    #‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, 
                    #‘sokalsneath’, ‘sqeuclidean’
    #**Note that ‘jensenshannon’,‘yule’may result in a condensed distance matrix which contains infinite value
    if 'L_metric' in hier_dict.keys():
        L_metric = hier_dict['L_metric']
   # L_method can be 'single', 'complete','average','weighted','centroid','median','ward'
    if 'L_method' in hier_dict.keys():
        L_method = hier_dict['L_method']
    if L_method == 'centroid' or L_method == 'median' or L_method == 'ward':
        if L_metric != 'euclidean':
            L_metric = 'euclidean'
            print('\n')
            print('*************Note:**************')
            print('Method '+str(L_method)+' requires the distance metric to be Euclidean')
        
    if 'optimal_ordering' in hier_dict.keys():
        optimal_ordering = hier_dict['optimal_ordering']
    else:
        optimal_ordering=False
    Z = linkage(x, method=L_method, metric=L_metric, optimal_ordering=optimal_ordering)
    #criterion can be 
    if 'criterionH' in hier_dict.keys():
        criterionH = hier_dict['criterionH']
    else:
        criterionH = 'inconsistent'
    if 'depth' in hier_dict.keys():
        depth = hier_dict['depth']
    else:
        depth = 2
    if 't' in hier_dict.keys():
        t = hier_dict['t']
        #for 'maxclust' or 'maxclust_monocrit' criteria,
         #t would be max number of clusters requested.
    elif criterionH == 'maxclust_monocrit' or criterionH == 'maxclust':
        t = 20

    if 'R' in hier_dict.keys():
        R = hier_dict['R']      
    if criterionH == 'inconsistent' or criterionH == 'maxclust_monocrit':
        #The inconsistency matrix to use for the 'inconsistent' criterion. 
        #R is computed if not provided.
        if R is None:
            R = inconsistent(Z, d=depth)
        else:
            R = np.asarray(R, order='c')
    if criterionH == 'monocrit':
        if R is None:
            R = inconsistent(Z, d=depth)
            #colR  is the column of 'R' to use as the statistic
        return fcluster(Z,criterion='monocrit',t=t, monocrit=maxRstat(Z, R, colR ))
    elif criterionH == 'maxclust_monocrit':
        return fcluster(Z,criterion='maxclust_monocrit',t=t, monocrit= maxinconsts(Z, R))
    else:
        return fcluster(Z,criterion=criterionH, depth=depth, R=R, t=t)

       
def get_clusters_kmeans_clustering(x,  kmeans_dict):
    #default value
    n_clusters=8
    init='k-means++'
    n_init=10
    max_iter=300
    tol=0.0001
    precompute_distances='auto'
    verbose=0
    random_state=None
    copy_x=True
    n_jobs=None
    algorithm='auto'
    if 'n_clusters' in kmeans_dict.keys():
        n_clusters = kmeans_dict['n_clusters']    
    if 'n_init' in kmeans_dict.keys():
        n_init = kmeans_dict['n_init']
    if 'init' in kmeans_dict.keys():
        init = kmeans_dict['init']
    if 'max_iter' in kmeans_dict.keys():
        max_iter = kmeans_dict['max_iter']
    if 'tol' in kmeans_dict.keys():
        tol = kmeans_dict['tol']
    if 'precompute_distances' in kmeans_dict.keys():
        precompute_distances = kmeans_dict['precompute_distances']
    if 'verbose' in kmeans_dict.keys():
        verbose = kmeans_dict['verbose']
    if 'random_state' in kmeans_dict.keys():
        random_state = kmeans_dict['random_state']
    if 'copy_x'in kmeans_dict.keys():     
        copy_x = kmeans_dict['copy_x']
    if 'n_jobs' in kmeans_dict.keys():
        n_jobs = kmeans_dict['n_jobs']
    if 'algorithm' in kmeans_dict.keys():
        algorithm = kmeans_dict['algorithm']
    return KMeans(n_clusters,init,n_init,max_iter,tol,precompute_distances,verbose,random_state,copy_x,n_jobs, algorithm).fit(x).labels_
    
    
    
def get_clusters_dbscan_clustering(x,dbscan_dict):
    #default value
    eps=0.5
    min_samples=5
    metric='euclidean'
    metric_params=None
    algorithm='auto'
    leaf_size=30
    p=None
    n_jobs=None
    if 'eps' in dbscan_dict.keys():
        eps = dbscan_dict['eps']    
    #else:
       # eps  = 0.5
    if 'min_samples' in dbscan_dict.keys():
        min_samples = dbscan_dict['min_samples']
    #else:
       # min_samples = 5
    if 'metric' in dbscan_dict.keys():
        metric = dbscan_dict['metric']
    #else:
        #metric = 'euclidean'
    if 'metric_params' in dbscan_dict.keys():
        metric_params = dbscan_dict['metric_params']
    if 'algorithm' in dbscan_dict.keys():
        algorithm = dbscan_dict['algorithm']
    if 'leaf_size' in dbscan_dict.keys():
        leaf_size = dbscan_dict['leaf_size']
    if 'p' in dbscan_dict.keys():
        p = dbscan_dict['p']
    if 'n_jobs' in dbscan_dict.keys():
        n_jobs = dbscan_dict['n_jobs']
    return DBSCAN(eps, min_samples, metric, metric_params, algorithm, leaf_size, p, n_jobs).fit(x).labels_    
    
    
def get_clusters_hdbscan_clustering(x,hdbscan_dict):
    #default value
    min_cluster_size_value = 5, 
    min_samples_value = 1,
    metric_value='euclidean', 
    alpha_value = 1.0,
    p_value = 2,
    algorithm_value = 'best', 
    leaf_size_value=40,
    cluster_selection_method_value = 'eom'

    #['best', 'generic', 'prims_kdtree', 'boruvka_kdtree','boruvka_balltree']
    if 'algorithm' in hdbscan_dict.keys():
        algorithm_value = hdbscan_dict['algorithm']    
    if 'alpha' in hdbscan_dict.keys():
        alpha_value = hdbscan_dict['alpha']
    #if 'gen_min_span_tree' in hdbscan_dict.keys():
        #gen_min_span_tree = hdbscan_dict['gen_min_span_tree']
    if 'leaf_size' in hdbscan_dict.keys():
        leaf_size_value = hdbscan_dict['leaf_size']

    if 'metric' in hdbscan_dict.keys():
        metric_value = hdbscan_dict['metric']
    if 'min_cluster_size' in hdbscan_dict.keys():
        min_cluster_size_value = hdbscan_dict['min_cluster_size']        
    if 'p' in hdbscan_dict.keys():
        p_value = hdbscan_dict['p']
    #['eom','leaf']
    if 'cluster_selection_method' in hdbscan_dict.keys():
        cluster_selection_method_value = hdbscan_dict['cluster_selection_method']
    if 'min_samples' in hdbscan_dict.keys():
        min_samples_value = int(hdbscan_dict['min_samples'])
    if metric_value == 'minkowski':
        p_value=2
    if algorithm_value == 'prims_kdtree':
        if metric_value not in KDTree.valid_metrics:
            print('Cannot use Prim\'s with KDTree for'+str(metric_value)+', change it to euclidean')
            metric_value = 'euclidean'
    if algorithm_value == 'boruvka_kdtree':
        if metric_value not in BallTree.valid_metrics:
            print('Cannot use Boruvka with KDTree for' +str(metric_value)+', change it to euclidean')
            metric_value ='euclidean'
    if algorithm_value == 'boruvka_balltree':
        if metric_value not in BallTree.valid_metrics:
            print('Cannot use Boruvka with BallTree for' +str(metric_value)+', change it to euclidean')
            metric_value='euclidean'
    if algorithm_value == 'boruvka_balltree':
        if metric_value not in BallTree.valid_metrics:
            print('Cannot use Boruvka with BallTree for' +str(metric_value)+', change it to euclidean')
            metric_value='euclidean'
    if algorithm_value == 'boruvka_balltree' and metric_value == 'sokalsneath':
        cluster_selection_method_value = 'minkowski'
        print('metric SokalSneathDistance is not valid for KDTree')
        #.astype(np.float64) is incase of the buffer dtype mismatch problem
    return hdbscan.HDBSCAN(min_cluster_size = min_cluster_size_value, min_samples = min_samples_value, 
                           metric = metric_value, alpha = alpha_value, p = p_value, 
                           algorithm = algorithm_value, leaf_size = leaf_size_value, memory=Memory(location=None), 
                           approx_min_span_tree=True, gen_min_span_tree=False, core_dist_n_jobs=4, 
                           cluster_selection_method = cluster_selection_method_value, allow_single_cluster=False, 
                           prediction_data=False, match_reference_implementation=False).fit(x.astype(np.float64)).labels_


def get_co_cluster(x, cell_names, ratio_resample=0.85, n_refeature=100,
                   round_resample=1000, knn=3,
                   algorithm='SNN', parameter_dict={'metric':'minkowski',
                                                    'method':'FastGreedy'}
                   ):
    # Initialization
    n_cells = x.shape[0]
    n_features = x.shape[1]
    co_occur = np.zeros([n_cells, n_cells])
    co_cluster = np.zeros([n_cells, n_cells])
    n_resample = int(n_cells*ratio_resample)
    n_refeature = int(min(n_refeature, n_features))
    print("Resample size: ", n_resample, n_refeature)
    # Steps for progress
    n_step = int(round_resample*0.1)

    # Iterations
    cur_co_cluster = np.zeros([n_cells, n_cells])
    diff_co_cluster = []
    for ct in range(round_resample):
        sample_lab = list(np.random.choice(np.arange(n_cells), n_resample, replace=False))
        feature_lab = list(np.random.choice(np.arange(n_features), n_refeature, replace=False))
        tp = x[sample_lab, :]
        if algorithm == 'SNN': # Other clustering methods: To be implemented
            metric = parameter_dict['metric']
            method = parameter_dict['method']
            if metric == "precomputed":
                tp = tp[:, sample_lab]
            else:
                tp = tp[:, feature_lab]
            clustering = get_clusters_SNN_community(tp, knn, metric, method)

        clusters = pd.DataFrame(clustering, index=sample_lab, columns=['cluster'])
        for i in sample_lab:
            for j in sample_lab:
                co_occur[i,j] = co_occur[i,j]+1
                if clusters.loc[i, 'cluster']==clusters.loc[j, 'cluster']:
                    co_cluster[i, j] = co_cluster[i, j] + 1
        diff = cur_co_cluster - (co_cluster+1) / (co_occur+1)
        diff_co_cluster.append(LA.norm(diff, 'fro'))
        cur_co_cluster = (co_cluster+1) / (co_occur+1)
        if ct%n_step ==0:
            print("Round ", ct+1, diff_co_cluster[-1])
    co_cluster = pd.DataFrame(cur_co_cluster, index=cell_names, columns=cell_names)

    # Plot 0: Show resample saturation
#     fig, ax = plt.subplots(1,1, figsize=(7,7))
#     plt.plot(diff_co_cluster, c=(0,0,0.8,0.5))
#     plt.yscale('log')
#     plt.xlabel('Iteration')
#     plt.ylabel('Fluctuation of Co-clusters')
#     fig.savefig('../Figure/Resample_saturation.pdf')
    return co_cluster

# def plot_co_cluster(co_cluster, save_prefix=None):
#     # Plot 1: Hierarchical clustering (by samples)
#
#     Z_sample = linkage(co_cluster, 'ward')
#     #     Z_sample = linkage(scipy.spatial.distance.squareform(1-co_cluster), 'ward')
#
#     thres = 10
#     fig, ax = plt.subplots(1, 1, figsize=(20, 8))
#     d = dendrogram(Z_sample, labels=co_cluster.index, leaf_rotation=90, leaf_font_size=10,
#                    orientation="top", color_threshold=None,
#                    )
#     # plt.axhline(y=thres, c='grey', lw=1, linestyle='dashed')
#     plt.title('Hierarchical Clustering Dendrogram')
#     plt.xlabel('sample index')
#     plt.ylabel('distance (Ward)')
#
#     # transforme the 'cyl' column in a categorical variable. It will allow to put one color on each level.
#     #     my_color=celltype.loc[d['ivl'], 'Sub_type'].cat.codes
#     my_color = [celltypes_col[CLA.meta_data.loc[i, "Subtype"]] for i in d['ivl']]  # replace with a new color map, to be implemented
#
#     # Apply the right color to each label
#     ax = plt.gca()
#     xlbls = ax.get_xmajorticklabels()
#     num = -1
#     for lbl in xlbls:
#         num += 1
#         lbl.set_color(my_color[num])
#     # if not save_prefix is None:
#     #     fig.savefig("../Figure/Dendrogram_AllNeurons_Resample.pdf")
#
#     # Plot 2: Heatmap
#     col_colors = pd.DataFrame({'Type': [celltypes_col[i] for i in CLA.meta_data["Subtype"]]},
#                               index=CLA.meta_data.index)
#     col_colors = col_colors.loc[co_cluster.index]
#     g = sns.clustermap(co_cluster, linecolor='white',
#                        row_colors=col_colors, col_colors=col_colors,
#                        row_linkage=Z_sample, col_linkage=Z_sample,
#                        annot=False, figsize=(12, 12))
#     # if save:
#     #     g.savefig("../Figure/Heatmap_CoCluster_AllNeurons.pdf")
#     return


def pickCLUSTERpara(method,selected_list):
    if len(selected_list) >0:
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
                                    _ = ns. get_clusters(method='Hierarchy',karg_dict=hier_dict)
                                    if len(selected_list)==0:
                                        selected_list = ns.metadata.index.tolist()
                                    tempARI = metrics.adjusted_rand_score(ns.metadata.loc[selected_list,'CellType'],
                                                                          ns.metadata.loc[selected_list,'Cluster'])
                                    tempDF = pd.DataFrame([tempARI, 
                                                           len(list(ns.metadata.groupby('Cluster'))),
                                                           str(hier_dict)]).T.copy()
                                    tempDF.columns=colname
                                    print(str(hier_dict))
                                    result_hier = result_hier.append(tempDF) 
                            elif criterionidx == 'distance':
                                _ = ns. get_clusters(method='Hierarchy',karg_dict=hier_dict)
                                tempARI = metrics.adjusted_rand_score(ns.metadata.loc[selected_list,'CellType'],
                                                                          ns.metadata.loc[selected_list,'Cluster'])
                                tempDF = pd.DataFrame([tempARI, 
                                                       len(list(ns.metadata.groupby('Cluster'))),
                                                       str(hier_dict)]).T.copy()
                                tempDF.columns=colname
                                print(str(hier_dict))
                                result_hier = result_hier.append(tempDF) 
                    if criterionidx == 'maxclust' or criterionidx == 'maxclust_monocrit':
                        for t_iter in  range(20,51): 
                            hier_dict.update(t = t_iter) 
                            _ = ns. get_clusters(method='Hierarchy',karg_dict=hier_dict)
                            tempARI = metrics.adjusted_rand_score(ns.metadata.loc[selected_list,'CellType'],
                                                                          ns.metadata.loc[selected_list,'Cluster'])
                            tempDF = pd.DataFrame([tempARI, 
                                                   len(list(ns.metadata.groupby('Cluster'))),
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
                     'n_jobs':12, 'algorithm':'auto'}
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
                                _ = ns. get_clusters(method='Kmeans',karg_dict=kmeans_dict)
                                tempARI = metrics.adjusted_rand_score(ns.metadata.loc[selected_list,'CellType'],
                                                                          ns.metadata.loc[selected_list,'Cluster'])
                                tempDF = pd.DataFrame([tempARI, 
                                                       len(list(ns.metadata.groupby('Cluster'))),
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
                     'leaf_size':30, 'p':None,'n_jobs':12}
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
                    p_iter =randrange(1,10)
                    dbscan_dict.update(p = p_iter)
                    while metric_iter == 'minkowski' and p_iter == 1:
                        p_iter =randrange(2,10)
                        dbscan_dict.update(p = p_iter)
                for epsidx in np.exp(-np.arange(0,4,0.5)):
                    dbscan_dict.update(eps = epsidx)
                    for min_samples_iter in range(5,10):
                        dbscan_dict.update(min_samples = min_samples_iter)
                        for leaf_size_iter in range(25,35):
                            dbscan_dict.update(leaf_size = leaf_size_iter)
                            _ = ns. get_clusters(method='DBSCAN',karg_dict=dbscan_dict)
                            tempARI = metrics.adjusted_rand_score(ns.metadata.loc[selected_list,'CellType'],
                                                                          ns.metadata.loc[selected_list,'Cluster'])
                            tempDF = pd.DataFrame([tempARI, len(list(ns.metadata.groupby('Cluster'))),str(dbscan_dict)]).T.copy()
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
                      'gen_min_span_tree':False,'core_dist_n_jobs':4,'cluster_selection_method':'eom',
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
                            _ = ns. get_clusters(method='HDBSCAN',karg_dict=hdbscan_dict)
                            tempARI = metrics.adjusted_rand_score(ns.metadata.loc[selected_list,'CellType'],
                                                                          ns.metadata.loc[selected_list,'Cluster'])
                            tempDF = pd.DataFrame([tempARI, len(list(ns.metadata.groupby('Cluster'))),str(hdbscan_dict)]).T.copy()
                            tempDF.columns = colname
                            print(str(hdbscan_dict))
                            result_hdbscan = result_hdbscan.append(tempDF)     

        idx_hdbscan = ['HDBSCAN'+str(x) for x in range(result_hdbscan.shape[0])]    
        result_hdbscan['idx'] = idx_hdbscan
        result_hdbscan.set_index('idx',inplace=True)       
        result_DF = result_hdbscan.copy()
    if method.lower() == 'snn':
        metric_list = [ 'euclidean','minkowski', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra',
                       'chebyshev','correlation','dice', 'hamming', 'jaccard','kulsinski', 'matching', 
                       'rogerstanimoto', 'russellrao','sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
        snn_dict = {'knn':5, 'metric':'minkowski','method':'FastGreedy'}
        result_snn= pd.DataFrame(columns = colname)
        for knn_iter in range(2,30):
            snn_dict.update(knn = knn_iter)
            for metric_idx in metric_list:
                snn_dict.update(metric = metric_idx)
                _ = ns. get_clusters(method = 'SNN_community',karg_dict = snn_dict)
                tempARI = metrics.adjusted_rand_score(ns.metadata.loc[selected_list,'CellType'],
                                                                          ns.metadata.loc[selected_list,'Cluster'])
                tempDF = pd.DataFrame([tempARI, len(list(ns.metadata.groupby('Cluster'))),str(snn_dict)]).T.copy()
                tempDF.columns = colname
                print(str(snn_dict))
                result_snn = result_snn.append(tempDF)
        idx_snn = ['SNN'+str(x) for x in range(result_snn.shape[0])]    
        result_snn['idx'] = idx_snn
        result_snn.set_index('idx',inplace=True)  
        result_DF = result_snn.copy()
    return result_DF.copy()