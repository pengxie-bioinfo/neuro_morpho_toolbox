import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import re
import pickle
from timeit import default_timer as timer
from sklearn.preprocessing import scale
from sklearn.manifold import Isomap, TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import scipy
import umap
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import sklearn.cluster
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu

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
        n_PC = min([df.shape[0], n_PC])
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
    return (g.community_fastgreedy(weights='weights').as_clustering().membership)


def get_clusters_Hierachy_clustering(x, karg_dict):
    #linkage(Z, method='single', metric='euclidean', optimal_ordering=False)
    if 'L_method' in karg_dict.keys():
        L_method = karg_dict['L_method']
        #L_metric can be braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, 
        #‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’,
        #‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
        #‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, 
        #‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
        if 'L_metric' in karg_dict.keys():
            L_metric = karg_dict['L_metric']
            Z = linkage(x,method=L_method,mtric = L_metric)
        else:
            Z = linkage(x,method=L_method)
    else:
        Z = linkage(x, method ='single', metric='euclidean')
    
    if 'criterion' in karg_dict.keys():
        criterionH = karg_dict['criterion']
        #fcluster(Z, numC=20, criterion='maxclust')
        if criterionH == 'maxclust':
            if 'numC' in karg_dict.keys():
                numC = karg_dict['numC']
            else:
                numC = 20
            return fcluster(Z,t=numC,criterion=criterionH)
        #fcluster(Z, copheneticD=0.9, criterion='distance')
        if criterionH == 'distance':
             if 'copheneticD' in karg_dict.keys():
                 copheneticD = karg_dict['copheneticD']
             else:
                 copheneticD = 0.9
             return fcluster(Z, t=copheneticD, criterion=criterionH)
        #fcluster(Z, t=0.9, depth=2, criterion='inconsistency')
        if criterionH == 'inconsistent':
             if 'depth' in karg_dict.keys():
                 depth = karg_dict['depth']
             else:
                 depth = 2
             if 't' in karg_dict.keys():
                 t = karg_dict['t']
             else:
                t = 0.9
             return fcluster(Z, t, depth ,criterion=criterionH)
    else:
        return fcluster(Z,t=0.9,criterion='inconsistent', depth=2, R=None, monocrit=None)

       
def get_clusters_kmeans_clustering(x,  n_clusters=20, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
    estimator= KMeans(n_clusters, random_state=100)
    return estimator.fit_predict(x,algorithm)
    
    
    
    
    
    
    
    
    
    
    
def plot_co_cluster(co_cluster, save_prefix=None):
    # Plot 1: Hierarchical clustering (by samples)

    Z_sample = linkage(co_cluster, 'ward')
    #     Z_sample = linkage(scipy.spatial.distance.squareform(1-co_cluster), 'ward')

    thres = 10
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    d = dendrogram(Z_sample, labels=co_cluster.index, leaf_rotation=90, leaf_font_size=10,
                   orientation="top", color_threshold=None,
                   )
    # plt.axhline(y=thres, c='grey', lw=1, linestyle='dashed')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance (Ward)')

    # transforme the 'cyl' column in a categorical variable. It will allow to put one color on each level.
    #     my_color=celltype.loc[d['ivl'], 'Sub_type'].cat.codes
    my_color = [celltypes_col[CLA.meta_data.loc[i, "Subtype"]] for i in d['ivl']]

    # Apply the right color to each label
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    num = -1
    for lbl in xlbls:
        num += 1
        lbl.set_color(my_color[num])
    # if not save_prefix is None:
    #     fig.savefig("../Figure/Dendrogram_AllNeurons_Resample.pdf")

    # Plot 2: Heatmap
    col_colors = pd.DataFrame({'Type': [celltypes_col[i] for i in CLA.meta_data["Subtype"]]},
                              index=CLA.meta_data.index)
    col_colors = col_colors.loc[co_cluster.index]
    g = sns.clustermap(co_cluster, linecolor='white',
                       row_colors=col_colors, col_colors=col_colors,
                       row_linkage=Z_sample, col_linkage=Z_sample,
                       annot=False, figsize=(12, 12))
    # if save:
    #     g.savefig("../Figure/Heatmap_CoCluster_AllNeurons.pdf")
    return


