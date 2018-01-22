import numpy as np
import math as math
from kmodes import kprototypes
import matplotlib.pyplot as plt

rows = 8535
cluster_start = 8
cluster_end = 21

syms = np.genfromtxt('merge_no_missing_values.csv', dtype=str, delimiter=',')[1:rows+1, 4]
X = np.genfromtxt('merge_no_missing_values.csv', dtype=object, delimiter=',')[1:rows+1, 1:]
X[:, 0] = X[:, 0].astype(float)
X[:, 1] = X[:, 1].astype(float)
X[:, 2] = X[:, 2].astype(float)

data_points =  X[:, 0:3].tolist()

def build_model_and_boxplot(no_of_clusters):
    kproto = kprototypes.KPrototypes(n_clusters=no_of_clusters, init='Cao')
    clusters = kproto.fit_predict(X, categorical=[3])
    # Print cluster centroids of the trained model.
    #print(kproto.cluster_centroids_[:1])

    distance = []

    for i in range(no_of_clusters):
        for j in range(i+1,no_of_clusters):
            distance.append(math.sqrt(math.pow(kproto.cluster_centroids_[0][i][0] - kproto.cluster_centroids_[0][j][0], 2) + 
                                         math.pow(kproto.cluster_centroids_[0][i][1] - kproto.cluster_centroids_[0][j][1], 2) +
                                         math.pow(kproto.cluster_centroids_[0][i][2] - kproto.cluster_centroids_[0][j][2], 2)))
    
    points_in_clusters = [[]]*no_of_clusters
    
    for k in range(rows):
        points_in_clusters[clusters[k]].append(data_points[k])
        
    max_distance_in_clusters = []
    
    for i in range(no_of_clusters):
        temp = []
        for j in points_in_clusters[i]:
            temp.append(math.sqrt(math.pow(kproto.cluster_centroids_[0][i][0] - j[0], 2) + 
                                         math.pow(kproto.cluster_centroids_[0][i][1] - j[1], 2) +
                                         math.pow(kproto.cluster_centroids_[0][i][2] - j[2], 2)))
        #Append the max distance between the points in the cluster  
        max_distance_in_clusters.append(max(temp))
    
    #print max_distance_in_clusters
    print "done",no_of_clusters


    return [distance, max_distance_in_clusters]

distance = []
density = []
for i in range(cluster_start, cluster_end):
    temp = build_model_and_boxplot(i)
    distance.append(temp[0])
    density.append(temp[1])
    
print "\nThe median of Distance is:"
for i in range(len(distance)):
    print "Median of {} is {}".format(cluster_start+i, np.median(distance[i]))
    
print "\nThe median of Density is:"
for i in range(len(distance)):
    print "Median of {} is {}".format(cluster_start+i, np.median(density[i]))

fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(distance)
ax.set_xticklabels([i for i in range(cluster_start, cluster_end)])
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.ylabel('Distance')
plt.xlabel('No of Clusters')
plt.show()

fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(density)
ax.set_xticklabels([i for i in range(cluster_start, cluster_end)])
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.ylabel('Density')
plt.xlabel('No of Clusters')
plt.show()


# Print training statistics
#print(kproto.cost_)
#print(kproto.n_iter_)

# for s, c in zip(syms, clusters):
#   print("Symbol: {}, cluster:{}".format(s, c))


