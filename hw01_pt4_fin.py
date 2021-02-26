
######## COLLABORATION ######
# I worked with Nick Mosca for this problem. We discussed the logic and approach together but we wrote our code separately
# %%
from __future__ import absolute_import

import random
import numpy as np

import matplotlib.pyplot as plt
import cv2

import scipy.io
import scipy.misc
# %%

def get_centroids(samples, clusters):
    """
    Find the centroid given the samples and their cluster.
    :param samples: samples.
    :param clusters: list of clusters corresponding to each sample.
    :return: an array of centroids.
    """
    cluster_label = np.unique(clusters)
    centroids = []

    for label in cluster_label:
        rows_cluster = [] #store pixel rows for each cluster
        for i in range(len(clusters)):
            if clusters[i] == label:
                rows_cluster.append(i)
            
        cluster_mean = np.mean(samples[rows_cluster,:], axis=0)
        centroids.append(cluster_mean.reshape(1,3))
    new_centroids = np.concatenate(centroids, axis=0)

    return new_centroids

# %%




def find_closest_centroids(samples, centroids):
    """
    Find the closest centroid for all samples.
    :param samples: samples.
    :param centroids: an array of centroids.
    :return: a list of cluster_id assignment.
    """

    results = []

    for row in range(samples.shape[0]):
        diff = samples[row] - centroids
        distance = (diff[:,0] **2 + diff[:,1] **2 + diff[:,2] **2) ** .5
        results.append(np.argmin(distance)) # return index - cluster assignment

    return results


def run_k_means(samples, initial_centroids, n_iter):
    """
    Run K-means algorithm. The number of clusters 'K' is defined by the size of initial_centroids


    :return: a pair of cluster assignment and history of centroids.
    """

    centroid_history = []
    current_centroids = initial_centroids #random centroid
    clusters = []
    for iteration in range(n_iter):
        centroid_history.append(current_centroids)
        print("Iteration %d, Finding centroids for all samples..." % iteration)
        clusters = find_closest_centroids(samples, current_centroids)
        print("Recompute centroids...")
        current_centroids = get_centroids(samples, clusters)

    return clusters, centroid_history
#import pdb; pdb.set_trace()

def choose_random_centroids(samples, K):
    """
    Randomly choose K centroids from samples.
    :param samples: samples.
    :param K: K as in K-means. Number of clusters.
    :return: an array of centroids.
    """
    shuffle_samples = np.random.permutation(samples)

    randos = shuffle_samples[0:K,:]
    
    return randos 

def main():

    '''This function takes in pixel data, 
    performs k-means clustering algorithm and returns image '''

    #load image 
    img = cv2.imread('boston-1993606_1280.jpg')
    depth, rows, columns = img.shape

    #reshape to 2 dimension array
    samples = img.reshape(depth*rows, columns)

    #choose random centroids, 10 clusters
    centroids = choose_random_centroids(samples,10)

    #k-means
    clusters, centroid_history = run_k_means(samples,centroids, n_iter=25)
    pixel_val = []

    #generate new image, loop cluster labels
    for label in clusters: 
        #replace cluster assignments with last iter centroids
        replace_centroid = np.array(centroid_history[-1][label]) 
        pixel_val.append(replace_centroid)
    new_img = np.concatenate(pixel_val)
    new_img_int = new_img.astype('uint8')
    final_img = new_img_int.reshape(depth, rows, columns)

    #image display
    cv2.imshow("image_output",final_img)
    cv2.imwrite('image_output.png',final_img) # writing new image
    cv2.waitKey(0) # just have to exit interactive window
    cv2.destroyAllWindow()




if __name__ == '__main__':
    random.seed(7)
    main()


