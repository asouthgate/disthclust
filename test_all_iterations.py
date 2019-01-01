import numpy as np
from multiprocessing import Process
from multiprocessing import Pool
import server3 as s3
import server3_test_functions as s3tf
from blockfilemmap import *
from linkage_functions import *
from worker3 import *
from itertools import product
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import hamming
from mylinke_single_euclidean import mylinkage
import time

n_workers = int(sys.argv[1])
n, d = map(int, sys.argv[2:4])
for i in range(100):

    # test with hamming distance,the setting can easily lead to distance ties, 
    #    n=np.random.randint(50,200)
    #    d=np.random.randint(20,100)
    X=np.random.randint(0,2,(n,d),dtype='uint8')
    print(n,d)

    #X = np.load('XX.npy')
    Z = s3tf.test_all1(X, "test_block_files", "test_data", n_workers)
    print("Server running, please run the worker")
    Z1 = mylinkage(X)

#    Z2 = linkage(X,method='complete',metric='euclidean')
#    Z2 = linkage(X,method='complete',metric=hamming_dist)

    print(Z)
    print(Z1)

    assert(np.all(Z-Z1[:,:3]<1e-3))
#    assert(np.all(Z-Z2[:,:3]<1e-3))
    print("passed test round!")
    print()
    time.sleep(5)

