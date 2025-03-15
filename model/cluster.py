from sklearn.cluster import KMeans
import joblib
import os
import numpy as np

def cluster(dataset_name, X, cluster_num=4, model_path='./model_info/clusters/'):
    """
    Construct the K-means clustering model to increase the complexity of discrimination
    :param dataset: the name of dataset
    :param dataset_dict: the dict of datasets
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :return: the K_means clustering model
    """
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    # if os.path.exists(model_path + dataset_name + '.pkl'):
    #     clf = joblib.load(model_path + dataset_name + '.pkl')
    # else:
    clf = KMeans(n_clusters=cluster_num, random_state=2019).fit(X)
    # joblib.dump(clf , model_path + dataset_name + '.pkl')
    return clf

def seed_test_input(clusters, limit):
    """
    Select the seed inputs for fairness testing
    :param clusters: the results of K-means clustering
    :param limit: the size of seed inputs wanted
    :return: a sequence of seed inputs
    """
    i = 0
    rows = []
    max_size = max([len(c[0]) for c in clusters])
    while i < max_size:
        if len(rows) == limit:
            break
        for c in clusters:
            if i >= len(c[0]):
                continue
            row = c[0][i]
            rows.append(row)
            if len(rows) == limit:
                break
        i += 1
    return np.array(rows)