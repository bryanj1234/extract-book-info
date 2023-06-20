from lib_book_corpus import *
from lib_keyword_and_ATE import *
import hashlib
from sklearn.cluster import KMeans as sklearn_KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.decomposition import FastICA as sklearn_FastICA
import sklearn.metrics as sklearn_metrics
import numpy as np
from cuml import UMAP as cuml_UMAP
import hdbscan
from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sentence_transformers.util import community_detection




#--------------------------------------------------------------------------------

# threshold = 0.99 is too high and dont' find anything much, 0.97 sees to be OK, start getting clutter at 0.95
def find_terms_closely_related_by_embedding_sim(vector_list, threshold = 0.96, min_community_size=2):
    community_list = community_detection(vector_list, threshold=threshold, min_community_size=min_community_size)

    return community_list

#--------------------------------------------------------------------------------

def vector_hash(vector):
    return hashlib.sha256(str(vector).encode()).hexdigest()

# NOTE: sklearn.preprocessing.normalize() returns a copy by default.
def l2_normalize(vectors):
    if vectors.ndim == 2:
        return normalize(vectors, copy=True)
    else:
        return normalize(vectors.reshape(1, -1), copy=True)[0]

# NOTE: Does nothing if len(vector_list) <= n_components
def UMAP_dimension_reduce(vector_list, n_neighbors, n_components, min_dist):
    if len(vector_list) <= n_components:
        return np.array(vector_list)

    umap_vectors = cuml_UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist).fit_transform(vector_list)

    return umap_vectors


def get_vector_mean(vectors):
    vectors = np.vstack([vector for vector in vectors])
    return vectors.mean(axis=0)


def get_normalized_vector_mean(vectors):
    return l2_normalize(get_vector_mean(vectors))


def get_inner_product_combinations(vec_list_1, vec_list_2):
    vec_list_1 = np.array(vec_list_1)
    vec_list_2 = np.array(vec_list_2)
    inner_product_cominations = np.einsum('ij, hj->ih', vec_list_1, vec_list_2)

    return inner_product_cominations


# Uses get_inner_product_combinations(), which returns inner products.
# So convert inner produc to similarity by sim = (1 + inner_product)/2
def get_vector_similarity(vec_1, vec_2, global_worst_inner_product = -1):
    inner_product = get_inner_product_combinations([vec_1], [vec_2])[0, 0]
    similarity = (inner_product - global_worst_inner_product)/(1 - global_worst_inner_product)

    return similarity


def get_global_worst_inner_product(vector_list):
    inner_product_cominations = get_inner_product_combinations(vector_list, vector_list)
    worst_similarity_by_vec = np.sort(inner_product_cominations)[:, 0]
    global_worst_inner_product = np.sort(worst_similarity_by_vec)[0]

    return global_worst_inner_product


def PCA_remove_excess_variance(vector_list, total_explained_variance_cutoff=0.70):

    # Try PCA.
    # See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    #
    #       "If 0 < n_components < 1 and svd_solver == 'full',
    #           select the number of components such that the amount of variance that
    #           needs to be explained is greater than the percentage specified by n_components."
    #
    pca = sklearn_PCA(n_components = total_explained_variance_cutoff, whiten=False).fit(vector_list)

    # Transform stuff to PCA component space
    PCA_space_vect_list = l2_normalize(pca.transform(vector_list))

    # New effective number of dimensions
    effective_num_dims = PCA_space_vect_list.shape[1]

    # Transform vectors to principle component space, then inverse transform so they have had low variance components removed...
    new_vector_list = l2_normalize(pca.inverse_transform(pca.transform(vector_list)))

    # Get PCA components in original space
    PCA_comp_list = l2_normalize(pca.components_)

    return new_vector_list, effective_num_dims, PCA_space_vect_list, PCA_comp_list


def group_vectors_by_cluster_label(vector_list, cluster_label_by_vec_index, label_list=False):
    if label_list is False:
        # Get list of labels
        label_list = list(set(cluster_label_by_vec_index))
        label_list.sort()

    idx_vecs_by_cluster_label = {}
    idx_vec_original_inds_by_cluster_label = {}
    for label in label_list:
        idx_vecs_by_cluster_label[label] = list(vector_list[ii] for ii in range(len(vector_list)) if cluster_label_by_vec_index[ii] == label)
        idx_vec_original_inds_by_cluster_label[label] = list(ii for ii in range(len(vector_list)) if cluster_label_by_vec_index[ii] == label)

    return idx_vecs_by_cluster_label, idx_vec_original_inds_by_cluster_label


#--------------------------------------------------------------------------------

def GAUSSIANMIXTURE_clustering(vector_list, n_components):
    n_neighbors = 15
    min_dist = 0.001
    umap_vectors = UMAP_dimension_reduce(vector_list, n_neighbors, n_components, min_dist)
    cluster_label_by_vec_index = GaussianMixture(n_components=n_components).fit_predict(np.array(umap_vectors))

    return cluster_label_by_vec_index


def AGGLOMERATIVE_clustering(vector_list, n_clusters):
    n_neighbors = 15
    n_components = 10
    min_dist = 0.001
    umap_vectors = UMAP_dimension_reduce(vector_list, n_neighbors, n_components, min_dist)
    cluster_label_by_vec_index = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(np.array(umap_vectors))

    return cluster_label_by_vec_index


def KMEANS_clustering(vector_list, n_clusters):
    n_neighbors = 15
    n_components = 10
    min_dist = 0.001
    umap_vectors = UMAP_dimension_reduce(vector_list, n_neighbors, n_components, min_dist)
    cluster_label_by_vec_index = sklearn_KMeans(n_clusters=n_clusters, random_state=0).fit_predict(np.array(umap_vectors))

    return cluster_label_by_vec_index


def HDBSCAN_clustering(vector_list,
                            min_cluster_size=15,
                            min_samples=2,
                            allow_single_cluster=True,
                            cluster_selection_method='leaf',
                            assign_noise_to_closest_cluster=True):
    n_neighbors = 15
    n_components = 10
    min_dist = 0.001
    umap_vectors = UMAP_dimension_reduce(vector_list, n_neighbors, n_components, min_dist)
    hdbscan_args = {'min_cluster_size':min_cluster_size,
                    'min_samples':min_samples,
                    'metric':'euclidean',
                    'cluster_selection_method':cluster_selection_method,
                    'allow_single_cluster':allow_single_cluster}
    clusterer = hdbscan.HDBSCAN(**hdbscan_args).fit(umap_vectors)
    cluster_label_by_vec_index = clusterer.labels_


    # Assign any noise vectors to cluster with closest mean, if desired.
    if assign_noise_to_closest_cluster:

        idx_vecs_by_cluster_label, _ = group_vectors_by_cluster_label(vector_list, cluster_label_by_vec_index)

        cluster_label_list = list(cluster_label for cluster_label in idx_vecs_by_cluster_label if cluster_label != -1)

        # Get means of (non-noise) clusters.
        cluster_mean_list = [get_normalized_vector_mean(idx_vecs_by_cluster_label[cluster_label]) for cluster_label in cluster_label_list]

        # Select noise vectors.
        noise_vectors_inds = list(ii for ii in range(len(cluster_label_by_vec_index)) if cluster_label_by_vec_index[ii] == -1)
        noise_vectors_list = list(vector_list[ii] for ii in range(len(cluster_label_by_vec_index)) if cluster_label_by_vec_index[ii] == -1)

        if len(noise_vectors_list) > 0 and len(cluster_mean_list) > 0:
            inner_product_combinations = get_inner_product_combinations(noise_vectors_list, cluster_mean_list)
            closest_cluster_mean_inds = np.argsort(-1*inner_product_combinations)[:,0]
            for ii in range(len(closest_cluster_mean_inds)):
                closest_cluster_label = cluster_label_list[closest_cluster_mean_inds[ii]]
                vector_ind = noise_vectors_inds[ii]
                # Set new cluster
                cluster_label_by_vec_index[vector_ind] = closest_cluster_label


    return cluster_label_by_vec_index


#--------------------------------------------------------------------------------

def plot_clusters(vector_list, cluster_label_by_vec_index, cluster_score, vector_silhouette_values, save_file=False):

    label_list = list(set(cluster_label_by_vec_index))
    label_list.sort()
    n_clusters = len(label_list)

    # Plot all Silhouette values ==================================================

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, len(vector_list) + (n_clusters + 1) * 10])

    y_lower = 10
    for label in label_list:
        label_cluster_silhouette_values = vector_silhouette_values[cluster_label_by_vec_index == label]
        label_cluster_silhouette_values.sort()
        size_label_cluster = label_cluster_silhouette_values.shape[0]

        y_upper = y_lower + size_label_cluster

        color = cm.nipy_spectral(float(label + 1) / n_clusters)

        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            label_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        text_x = -1 if label %2 == 0 else -.7
        ax1.text(text_x, y_lower + 0.5 * size_label_cluster, str(label))

        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette plot, average score = " + str(round(cluster_score, 2)))
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=cluster_score, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # Cluster scatterplot ===========================================================

    # First reduce to 5 dimensions and normalize. Hopefully will improve scale of resulting plot.
    n_neighbors = 15
    n_components = 10
    min_dist = 0.001
    prelim_reduced_data = l2_normalize(UMAP_dimension_reduce(vector_list, n_neighbors, n_components, min_dist))
    # Reduce to 2 dimensions...
    n_neighbors = 15
    n_components = 2
    min_dist = 0.0
    umap_data = UMAP_dimension_reduce(prelim_reduced_data, n_neighbors, n_components, min_dist)

    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = cluster_label_by_vec_index

    colors = [cm.nipy_spectral(float(label + 1) / n_clusters) for label in cluster_label_by_vec_index]
    ax2.scatter(result.x, result.y, color=colors, s=5.0)

    ax2.set_title("Scatterplot of clusters")

    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()


#--------------------------------------------------------------------------------

def recursively_print_cluster_tree(cluster_rec):
    level = cluster_rec['level']
    print("\n")
    print("\t"*level, "#####################################################################")
    print("\t"*level, "cluster_mean_sim_to_global_mean:", cluster_rec['mean_vec_sim_to_global_mean'])
    print("\t"*level, "=====================================================================")
    print("\t"*level, "EXEMPLAR:", cluster_rec['exemplar_ind'], cluster_rec['exemplar_desc'].replace("\n", " ")[:80])
    print("\t"*level, "sim_to_cluster_mean:", cluster_rec['exemplar_sim_to_cluster_mean'], "sim_to_global_mean:", cluster_rec['exemplar_sim_to_global_mean'])
    print("\t"*level, "=====================================================================")
    print("\t"*level, "GLOBAL BEST:", cluster_rec['global_best_ind'], cluster_rec['global_best_desc'].replace("\n", " ")[:80])
    print("\t"*level, "sim_to_cluster_mean:", cluster_rec['global_best_to_cluster_mean'], "sim_to_global_mean:", cluster_rec['global_best_to_global_mean'])
    print("\t"*level, "=====================================================================")


    if 'addl_item_ind_for_mean' in cluster_rec:
        print("\t"*level, "=====================================================================")
        print("\t"*level, "ADDTIONAL ITEM FOR MEAN:",
                            cluster_rec['addl_item_ind_for_mean'],
                            cluster_rec['addl_item_desc_for_mean'].replace("\n", " ")[:80])
        print("\t"*level, "=====================================================================")
        print("\t"*level, "ADDTIONAL ITEM FOR EXEMPLAR:",
                            cluster_rec['addl_item_ind_for_exemplar'],
                            cluster_rec['addl_item_desc_for_exemplar'].replace("\n", " ")[:80])
        print("\t"*level, "=====================================================================")
        print("\t"*level, "ADDTIONAL ITEM FOR GLOBAL BEST:",
                            cluster_rec['addl_item_ind_for_global_best'],
                            cluster_rec['addl_item_desc_for_global_best'].replace("\n", " ")[:80])
        print("\t"*level, "=====================================================================")

    print("\t"*level, cluster_rec['cluster_id'],
            cluster_rec['parent_id'],
            'sim % to parent:', str(100 * cluster_rec['parent_similarity']),
            "# vectors:", cluster_rec['num_vectors'],
            "scores:", cluster_rec['cluster_score'], cluster_rec['davies_bouldin_score'], cluster_rec['davies_bouldin_score']
        )
    for child_cluster_rec in cluster_rec['children']:
        recursively_print_cluster_tree(child_cluster_rec)


# Finds the vector in the cluster closest to the cluster mean.
# Also, compute the similarities of the child cluster mean to the parent cluster mean.
def recursively_add_cluster_exemplars(cluster_rec, vector_index_by_vector_hash, vector_descriptions_list, global_mean_vec=False):
    if global_mean_vec is False:
        global_mean_vec = cluster_rec['mean_vec']

    cluster_rec['mean_vec_sim_to_global_mean'] = get_vector_similarity(cluster_rec['mean_vec'], global_mean_vec)

    inner_product_combinations = get_inner_product_combinations([cluster_rec['mean_vec']], cluster_rec['cluster_vectors'])
    exemplar_local_ind = np.flip(np.argsort(inner_product_combinations))[0,0]
    exemplar_vec = cluster_rec['cluster_vectors'][exemplar_local_ind]
    exemplar_hash = vector_hash(exemplar_vec)
    cluster_rec['exemplar_ind'] = vector_index_by_vector_hash[exemplar_hash]
    cluster_rec['exemplar_vec'] = exemplar_vec
    cluster_rec['exemplar_desc'] = vector_descriptions_list[cluster_rec['exemplar_ind']]
    cluster_rec['exemplar_sim_to_cluster_mean'] = get_vector_similarity(cluster_rec['exemplar_vec'], cluster_rec['mean_vec'])
    cluster_rec['exemplar_sim_to_global_mean'] = get_vector_similarity(cluster_rec['exemplar_vec'], global_mean_vec)

    inner_product_combinations = get_inner_product_combinations([global_mean_vec], cluster_rec['cluster_vectors'])
    global_best_local_ind = np.flip(np.argsort(inner_product_combinations))[0,0]
    global_best_vec = cluster_rec['cluster_vectors'][global_best_local_ind]
    global_best_hash = vector_hash(global_best_vec)
    cluster_rec['global_best_ind'] = vector_index_by_vector_hash[global_best_hash]
    cluster_rec['global_best_vec'] = global_best_vec
    cluster_rec['global_best_desc'] = vector_descriptions_list[cluster_rec['global_best_ind']]
    cluster_rec['global_best_to_cluster_mean'] = get_vector_similarity(cluster_rec['global_best_vec'], cluster_rec['mean_vec'])
    cluster_rec['global_best_to_global_mean'] = get_vector_similarity(cluster_rec['global_best_vec'], global_mean_vec)

    cluster_rec['parent_similarity'] = -1 # Override later if winds up being the child of something.

    for child_cluster_rec in cluster_rec['children']:
        recursively_add_cluster_exemplars(child_cluster_rec,
                                            vector_index_by_vector_hash,
                                            vector_descriptions_list,
                                            global_mean_vec=global_mean_vec)
        child_cluster_rec['parent_similarity'] = round(get_vector_similarity(cluster_rec['mean_vec'], child_cluster_rec['mean_vec']), 4)


def recursively_cluster(cluster_rec, n_sub_clusters, max_depth=3):
    cluster_id = cluster_rec['cluster_id']
    level = cluster_rec['level']


    #print("\t"*level, cluster_id, len(cluster_rec['cluster_vectors']))

    cluster_rec['num_vectors'] = len(cluster_rec['cluster_vectors'])

    # Get mean of the vectors in the input cluster.
    cluster_rec['mean_vec'] = get_normalized_vector_mean(cluster_rec['cluster_vectors'])

    # Nothing to do if # of vectors is not greater than n_sub_clusters
    if level >= max_depth or len(cluster_rec['cluster_vectors']) <= n_sub_clusters:
        cluster_rec['cluster_score']  = False
        cluster_rec['davies_bouldin_score'] = False
        cluster_rec['calinski_harabasz_score'] = False
        return cluster_id + 1


    # Cluster the input cluster into sub-clusters.

    #AGGLOMERATIVE_clustering sees to work the best of the fixed-number-of components clustering, and is more predicatble than HDBSCAN

    cluster_label_by_vec_index = AGGLOMERATIVE_clustering(cluster_rec['cluster_vectors'], n_sub_clusters) # FIXME: CUDA sometimes runs out of memory here.

    #cluster_label_by_vec_index = KMEANS_clustering(cluster_rec['cluster_vectors'], n_sub_clusters)
    #cluster_label_by_vec_index = GAUSSIANMIXTURE_clustering(cluster_rec['cluster_vectors'], n_sub_clusters) # FIXME: SOmetimes blows up, need to reslove.
    # cluster_label_by_vec_index = HDBSCAN_clustering(cluster_rec['cluster_vectors'],
    #                                             min_cluster_size=max(      int( len(cluster_rec['cluster_vectors'])/(3*n_sub_clusters)), 2)
    #                                         )

    #cluster_label_by_vec_index = HDBSCAN_clustering(cluster_rec['cluster_vectors'], cluster_selection_method='leaf') # FIXME: CUDA sometimes runs out of memory here.
    # Only needed if using HDBSCAN_clustering()
    # if len(set(cluster_label_by_vec_index)) <= 2:
    #     cluster_rec['cluster_score']  = False
    #     cluster_rec['davies_bouldin_score'] = False
    #     cluster_rec['calinski_harabasz_score'] = False
    #     return cluster_id + 1

    cluster_rec['cluster_score']  = sklearn_metrics.silhouette_score(cluster_rec['cluster_vectors'], cluster_label_by_vec_index)
    cluster_rec['davies_bouldin_score'] = sklearn_metrics.davies_bouldin_score(cluster_rec['cluster_vectors'], cluster_label_by_vec_index)
    cluster_rec['calinski_harabasz_score'] = sklearn_metrics.calinski_harabasz_score(cluster_rec['cluster_vectors'], cluster_label_by_vec_index)

    idx_vecs_by_cluster_label, _ = group_vectors_by_cluster_label(cluster_rec['cluster_vectors'], cluster_label_by_vec_index)

    parent_id = cluster_id
    cluster_id += 1
    level += 1
    for cluster_label in idx_vecs_by_cluster_label:
        cluster_vectors = idx_vecs_by_cluster_label[cluster_label]
        child_cluster_rec = {
            'cluster_id':cluster_id,
            'parent_id':parent_id,
            'level':level,
            'cluster_vectors':np.array(cluster_vectors),
            'children':[]
        }
        cluster_rec['children'].append(child_cluster_rec)

        # Cluster the child rec
        cluster_id = recursively_cluster(child_cluster_rec, n_sub_clusters, max_depth=max_depth)

    return cluster_id


def recursively_cluster_and_add_exemplars(cluster_rec,
                                            n_sub_clusters,
                                            vector_index_by_vector_hash,
                                            vector_descriptions_list,
                                            max_depth=3):
    print("Recursively clustering...")
    recursively_cluster(cluster_rec, n_sub_clusters, max_depth=max_depth)
    print("Adding exemplars...")
    recursively_add_cluster_exemplars(cluster_rec, vector_index_by_vector_hash, vector_descriptions_list)


def recursively_add_additional_closest_item_desc(cluster_rec, item_descriptions_list, item_list):
    # Get best additional item for cluster mean vector
    inner_product_combinations = get_inner_product_combinations([cluster_rec['mean_vec']], item_list)
    addl_item_ind = np.flip(np.argsort(inner_product_combinations))[0,0]
    addl_item_desc = item_descriptions_list[addl_item_ind]
    cluster_rec['addl_item_ind_for_mean'] = addl_item_ind
    cluster_rec['addl_item_desc_for_mean'] = addl_item_desc
    cluster_rec['addl_item_vec_for_mean'] = item_list[addl_item_ind]

    # Get best additional item for cluster exemplar vector
    inner_product_combinations = get_inner_product_combinations([cluster_rec['exemplar_vec']], item_list)
    addl_item_ind = np.flip(np.argsort(inner_product_combinations))[0,0]
    addl_item_desc = item_descriptions_list[addl_item_ind]
    cluster_rec['addl_item_ind_for_exemplar'] = addl_item_ind
    cluster_rec['addl_item_desc_for_exemplar'] = addl_item_desc
    cluster_rec['addl_item_vec_for_exemplar'] = item_list[addl_item_ind]

    # Get best additional item for global_best_vec
    inner_product_combinations = get_inner_product_combinations([cluster_rec['global_best_vec']], item_list)
    addl_item_ind = np.flip(np.argsort(inner_product_combinations))[0,0]
    addl_item_desc = item_descriptions_list[addl_item_ind]
    cluster_rec['addl_item_ind_for_global_best'] = addl_item_ind
    cluster_rec['addl_item_desc_for_global_best'] = addl_item_desc
    cluster_rec['addl_item_vec_for_global_best'] = item_list[addl_item_ind]

    for child_cluster_rec in cluster_rec['children']:
        recursively_add_additional_closest_item_desc(child_cluster_rec, item_descriptions_list, item_list)


def recursively_flatten_cluster_tree(cluster_rec, cluster_list):
    cluster_list.append(cluster_rec)
    for child_cluster_rec in cluster_rec['children']:
        recursively_flatten_cluster_tree(child_cluster_rec, cluster_list)

def get_cluster_leaves(cluster_list):
    leaves_list = []
    for rec in cluster_list:
        if len(rec['children']) == 0:
            leaves_list.append(rec)

    return leaves_list

def get_cluster_path(cluster_id, cluster_list):
    cur_cluster = cluster_list[cluster_id]
    cluster_path = [cur_cluster['cluster_id']]
    while cur_cluster['parent_id'] is not False:
        cur_cluster = cluster_list[cur_cluster['parent_id']]
        cluster_path.append(cur_cluster['cluster_id'])

    return cluster_path

#--------------------------------------------------------------------------------