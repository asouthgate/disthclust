import random, time
import socket
import numpy as np
from multiprocessing.managers import BaseManager
from update_map import UpdateMap
from queue import Queue
from linkage_functions import *

class WorkerList():
    def __init__(self):
        self.workers = set()
    def add(self,wi):
        self.workers.add(wi)
    def remove(self,wi):
        del self.workers[wi]
    def get_worker_ids(self):
        return self.workers
    def get_n_workers(self):
        return len(self.workers)

class QueueManager(BaseManager):
    pass

# ******************** DOES NOT NEED A NTASK
# ******************** CAN CHECK IF QUEUE IS EMPTY YET
globalTaskQueue = Queue()
globalResultQueue = Queue()
globalUpdateMap = UpdateMap()
workers = WorkerList()

class BlockServer():
    def __init__(self, n_workers):

        # Register the queue in the network
        QueueManager.register('get_gtask_queue', callable=lambda: globalTaskQueue)
        QueueManager.register('get_gres_queue', callable=lambda: globalResultQueue)
        QueueManager.register('get_gup_map', callable=lambda: globalUpdateMap)
        QueueManager.register('get_workers', callable=lambda: workers)

        gPort=5000
        gKey=b'baps'
        self.gManager = QueueManager(address=('', gPort), authkey=gKey)

        # Check failure to start?
        self.gManager.start()
        print("server estabished at: " + socket.gethostname())

        # Keep a list of current workers connected
        self.workers = self.gManager.get_workers()

        # Obtain queue from the network
        self.globalTaskQueue = self.gManager.get_gtask_queue()
        self.globalResultQueue = self.gManager.get_gres_queue()
        self.globalUpdateMap = self.gManager.get_gup_map()
        self.workers = self.gManager.get_workers()

        # Simple counter
        self.nTask = 0
        
        print("Waiting for workers to connect")
        # Block until all workers are connected
        while self.workers.get_n_workers() < n_workers:
            pass
        self.worker_ids = sorted(list(self.workers.get_worker_ids()))

    def update_workers(self, updateName, *params):
        print("Server updating workers", updateName)
        for wid in self.worker_ids:
            self.globalUpdateMap.put(wid, updateName, *params)
        print("Server done updating")

    def collect_updates(self):
        print("Server collecting updates")
        while not self.globalUpdateMap.is_empty():
            pass
        print("Updates collected")

    def submit_task(self, funcName, *params):
        # Submit a task to the queue
        print("Server submitting", funcName, self.nTask)
        self.globalTaskQueue.put((funcName, self.nTask, *params))
        self.nTask += 1

    def collect(self): 
        # Collect values returned by workers
        print("Collecting. Waiting for the workers...!", self.globalTaskQueue)
        results = []
#        while not self.globalTaskQueue.empty():
        while self.nTask > 0:
            # Block manually, that way we can get the fastest ones first
            result = self.globalResultQueue.get()    
            print("result", result)
            if result:
                ind = result
                # Convert it into the correct data format
                print('Collecting %d, result:', result)
                results.append(ind)
                self.nTask -=1
        return results

    def shutdown(self):
        print("shutting down")
        time.sleep(10)
        gManager.shutdown()

def split_and_write_data(X, data_folder, nb, bs):
    # Splits data X into nb blocks
    # This is not the matrix, but a 1D array
    for b in range(nb):
        bi = bs * b
        fn =  "%s/%d.npy" % (data_folder, b)
        np.save(fn, X[bi:bi+bs])

def linkage_block(X, base_directory, data_folder):
    # Establish server
    serv = BlockServer()

    # Prepare data
    n,d=X.shape
    constants.init(n,d)
    init_files(base_directory, constants.N_BLOCK)
    hedInd = np.zeros(n-1,dtype=constants.DATA_TYPE)
    hedVal = np.zeros(n-1,dtype=constants.DATA_TYPE) 
 
    beditPrev = editPool()
    beditNext = editPool()
    
    nodeFlag = np.zeros(constants.N_BLOCK*constants.BLOCK_SIZE, dtype=bool)
    nodeFlag[:n]=True
    blockFlag = np.ones(constants.N_BLOCK)>0
    blockCount = np.zeros(constants.N_BLOCK)+constants.BLOCK_SIZE
    if constants.N_NODE%constants.BLOCK_SIZE!=0:
        blockCount[-1]=constants.N_NODE%constants.BLOCK_SIZE
           
    bprev = np.zeros((constants.N_BLOCK,constants.N_BLOCK),dtype=object)
    bnext = np.zeros((constants.N_BLOCK,constants.N_BLOCK),dtype=object)
    bdist = np.zeros((constants.N_BLOCK,constants.N_BLOCK),dtype=object)

    # Split X into blocks, saving each as a numpy data file
    split_and_write_data(X, data_folder, constants.N_BLOCK, constants.BLOCK_SIZE)

    # Task 1: Compute distMat
    for bi in range(0,constants.N_BLOCK):
        for bj in range(bi,constants.N_BLOCK):
        # Init dist block
            serv.submit_task("cal_dist", bi, bj)
    serv.collect()

    # Task 2: Sort block rows in parallel
    nb = constants.N_BLOCK
    bs = constants.BLOCK_SIZE

    for bi in range(0, constants.N_BLOCK):
        serv.submit_task("sort_rows", bi, blockFlag)
    serv.collect()

    # Insert the results back into hedInd, hedVal
    for bi, subHedInd, subHedVal in res:
        mil = bi * constants.BLOCK_SIZE
        mir = (bi+1) * constants.BLOCK_SIZE
        if mir < len(hedInd):
            hedInd[mil:mir] = subHedInd
            hedVal[mil:mir] = subHedVal
        else:
            hedInd[mil:] = subHedInd
            hedVal[mil:] = subHedVal

    # Core algorithm:
    treeNodeArr=np.arange(constants.N_NODE,dtype=constants.DATA_TYPE)
    Z = np.zeros((constants.N_NODE-1,3), dtype='float')        
    for iStep in range(constants.N_NODE-1):
        # First find ii, jj to be merged
        minind, minval = constants.mymin(hedVal) 
        ii = minind
        jj = hedInd[ii] 
        assert(jj>ii)
        assert(nodeFlag[jj])        

        Z[iStep,0:2] = np.sort(treeNodeArr[[ii,jj]])
        Z[iStep,2] = minval
        
        # Next update pairwise distances of ii to others nodes        
        # Extract the Lvector of ii and jj, get maxes etc
        nodeFlag[jj]=False
        update_pair_dist(bdist, nodeFlag, blockFlag, ii, jj)       
        treeNodeArr[ii] = iStep+constants.N_NODE
        treeNodeArr[jj] = 0        
        nodeFlag[jj]=True
        
        # Next, to parallelize:
        [bii, iii] = constants.getbi(ii);
        [bjj, jjj] = constants.getbi(jj);    

        # Compute each row in parallel
        print(0, bjj+1)
        for bk in range(0, bjj+1):
            bkl = bk * constants.BLOCK_SIZE
            bkr = (bk+1) * constants.BLOCK_SIZE
            if bkr < len(hedInd):
                serv.submit_task("recalc_blocks", bk, ii, jj, hedInd[bkl:bkr], hedVal[bkl:bkr])
            else:
                serv.submit_task("recalc_blocks", bk, ii, jj, hedInd[bkl:], hedVal[bkl:])
        print("collecting...")
        res = serv.collect()
        print("result", res)
        for bi, subHedInd, subHedVal in res:
            mil = bi * constants.BLOCK_SIZE
            mir = (bi+1) * constants.BLOCK_SIZE
            if mir < len(hedInd):
                hedInd[mil:mir] = subHedInd
                hedVal[mil:mir] = subHedVal
            else:
                hedInd[mil:] = subHedInd
                hedVal[mil:] = subHedVal

        # Update the workers
        serv.update_workers("update_nodeflag", jj)
        serv.globalUpdateMap.collect_updates()
        serv.update_workers("update_blockflag", bjj)
        serv.globalUpdateMap.collect_updates()

        # Update the flags here too
        nodeFlag[jj]=False
        if jj<constants.N_NODE-1:
            hedInd[jj]=constants.DEL_VAL
            hedVal[jj]=constants.DEL_VAL
        
        blockCount[bjj] -= 1
        blockFlag[bjj] = blockCount[bjj]>0

    return Z


if __name__ == "__main__":
    bs = BlockServer()
    bs.submit_task("f",10,1)
    print("SUBMITTED, COLLECTING")
    bs.collect()
    print("NEXT")
    time.sleep(10)
    bs.submit_task("f",10, 1)
    bs.collect()
    print("FINISHED")
    
