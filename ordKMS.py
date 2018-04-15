import math
import csv
import time
import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score


#===================================================Euclidian distance=====================
def eucl_distance(point_one, point_two):                  #define function to measure Euclidian distance between point_one and point_two
	if(len(point_one) != len(point_two)):                 #check for an error (len(1st point)is not equal len(2nd point)) 
		raise Exception("Error: non comparable points")   #check for an error 

	sum_diff = 0.0                                                 #set variable sum_diff
	for i in range(len(point_one)):                                #iterating take difference between each elem from 1st and 2nd point
		diff = pow((float(point_one[i]) - float(point_two[i])), 2) #and square this difference. 
		sum_diff += diff                                           #As the result summing all these differencess and equating to sum_diff
	final = math.sqrt(sum_diff)                                    #sqrt from sum_diff will be Euqlidian dist betw(1st and 2nd points)
	return final
#===================================================Devide data set to further scattering=====================

global dimensions, num_clusters, num_points,dimensions,data,flag

 										#turn on a timer which allows us to estimate performane of this algorithm
#===============================================reading and preparing data set======================
print("Enter the number of clusters you want to make: ")
num_clusters = input()
num_clusters = int(num_clusters)
start_time = time.time()
with open('3D_spatial_network.csv','rb') as f:
    reader = csv.reader(f)
    data = list(reader)

data.pop(0)
for i in range (len(data)):
    data[i].pop(0)
data=np.array(data).astype(np.float)
data=data[0:10000]
#Print(data)
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data).labels_
# 	print('data',[ data[i] for i in [indices] ])
# 	data=np.array ([[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]])
#====================================================================================================

#================================================Initialize centroids matrix=========================
initial=[]
for i in xrange(num_clusters):
    initial.append(data[i])
initial=np.vstack(initial)
#====================================================================================================

num_points = len(data)                                    #number of rows
dimensions = len(data[0])                                 #number of columns
#chunks = [ [] for _ in range(size) ]

#for i, chunk in enumerate(data):
#	chunks[i % size].append(chunk)

#====================================================================================================

flag= True
while flag==True:

	cluster=[]
	#print str(rank) + ': ' + str(data)
	#===================================Calculating dist matrix in each process==============================
	dist =np.zeros((len(data),len(initial)))


	for j in range(len(initial)):
		for i in range(len(data)):
			dist[i][j]=np.linalg.norm(initial[j]-data[i])
	#print('dist',dist)		
	#===================================Initilize lable for each sample in each process======================
	for i in range (len(dist)):										#iterable take each raw in dist matrix and 
		cluster.append(np.argmin(dist[i])+1)                       #find column index of min value (this index is number of centorud)
	#print('clust vect',cluster)
	#===================================Calculating the number of samples in each cluster====================
	Q_clusts=collections.Counter(cluster)							
	#Q_clusts=np.array((collections.Counter(clusters).keys(),collections.Counter(clusters).values()))# what is the labels was faced in each process and their frequency 

	#=========================================================================================================
	#====================================From each worker we gather cluster vector and join them ============
	#==========================================================================================================
	#if rank==0:
	#	cluster=[item for sublist in cluster for item in sublist]
	#	data = [item.tolist() for item in data]
	#	data=[item for sublist in data for item in sublist]
	#	print(data)
	#	print(cluster)

	centroid=np.zeros((len(initial),len(initial[0])))
	for k in range (1,num_clusters+1):
		indices = [i for i, j in enumerate(cluster) if j == k]
		#print('ind',indices)        
		#print(k,  [data[i] for i in indices] )
		#print('sum',np.sum([data[i] for i in indices], axis=0 ))
		#print('div',np.divide((np.sum([data[i] for i in indices], axis=0)).astype(np.float),totcounter[k]))
		centroid[k-1]=np.divide((np.sum([data[i] for i in indices], axis=0)).astype(np.float),Q_clusts[k])


	#print('centroids',centroids)
	#print ('initial', initial)
	#print('centroid',centroid)
		
	
	if np.all(centroid==initial):
		flag=False
		
		print ("Execution time %s seconds" % (time.time() - start_time))
	else:
		#print ('initial after', initial)
		initial= centroid 
	
	#print('final clust vect',cluster)
	#print('libkms clust vect',kmeans)
	print('adjusted_rand_score',adjusted_rand_score(kmeans,cluster))

