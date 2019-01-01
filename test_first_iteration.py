import numpy as np
from multiprocessing import Process
from multiprocessing import Pool
import server3 as s3
import server3_test_functions as s3tf
from blockfilemmap import *
from linkage_functions import *
from worker3 import *
from itertools import product

n, d = map(int, sys.argv[1:3])
X = np.random.randint(0,2,(n,d),dtype='uint8')
print(X)
print(n, d, "test_block_files", "test_data")


constants.init(n,d)
nb, bs = constants.N_BLOCK, constants.BLOCK_SIZE
trueTNA, truehedInd, truehedVal = s3tf.test_core2(X, "test_block_files")
s3_TNA, s3_hedInd, s3_hedVal = s3tf.test_core1(X, "test_block_files", "test_data")
print("Test: server initialized, please run workers separately")

assert len(s3_hedInd) == len(s3_hedVal)
print("asserting hed, inds")
for i in range(len(s3_hedInd)):
    print(i)
    print(s3_hedInd[i], truehedInd[i])
    print(s3_hedVal[i], truehedVal[i])
    print()
    assert s3_hedInd[i] == truehedInd[i]
    assert s3_hedVal[i] == truehedVal[i]

for i in range(len(trueTNA)):
    assert trueTNA[i] == s3_TNA[i]

print(s3_hedInd)
print(truehedInd)
print()
print(s3_hedVal)
print(truehedVal)

print("Core first iteration test passed!")
