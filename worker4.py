import time, sys, socket, random
import numpy as np
from multiprocessing import Pool
from linkage_functions import *
import multiprocessing
from multiprocessing import cpu_count
from multiprocessing.managers import BaseManager, NamespaceProxy
from update_map import UpdateMap
from blockfilemmap import BlockFileMap
import common_base as cb


def cal_dist(self, bi, bj):
    """ 
    Takes a block index bi, bj
    Calculates the pairwise distances for the block
    """
    assert bj >= bi
    # Load saved file
    subXi = np.load("%s/%d.npy" % (data_directory, bi))
    subXj = np.load("%s/%d.npy" % (data_directory, bj))

    dist_block_arr = np.zeros(shape=(constants.BLOCK_SIZE,constants.BLOCK_SIZE))

    tmpargs = []
    tmpindi = []
    tmpindj = []

    # Calculate elements; different indices to cal depending on bi, bj
    # Also edge cases
    if bj > bi:
        ind = (np.arange(constants.BLOCK_SIZE), np.arange(constants.BLOCK_SIZE))
        for i in range(len(subXi)):
            for j in range(len((subXj))):
                if not (i>=constants.N_NODE or j>=constants.N_NODE):
                    xi, xj = subXi[i], subXj[j]
                    tmpargs.append((xi, xj))
                    tmpindi.append(i)
                    tmpindj.append(j)

    elif bi == bj:
        for i in range(len(subXi)-1):
            for j in range(i+1, len((subXj))):
                if not (i>=constants.N_NODE or j>=constants.N_NODE):
                    xi, xj = subXi[i], subXj[j]
                    tmpargs.append((xi, xj))
                    tmpindi.append(i)
                    tmpindj.append(j)

    with multiprocessing.Pool(processes=self.nCores) as pool:
        results = pool.starmap(cal_dist_ij, tmpargs)

    print(tmpindi, tmpindj)
#        assert (dist_block_arr[tmpindi, tmpindj] == results).all()
    dist_block_arr[tmpindi, tmpindj] = results

    # Pad with np.nans
    # Write the result
    bfd = "%s/%d_d/%d_d.block" % (block_directory, bi, bj)
    dist_block = BlockFileMap(bfd, constants.DATA_TYPE, dist_block_arr.shape)
    dist_block.open()
    dist_block.write_all(dist_block_arr)
    dist_block.close()
    return bi, bj

def sort_rows(self, bi):
    """
    Takes a row index bi, and a subset of hedInd, hedval
    And sorts the row 
    """
    # Create subHedInd, subHedVal
    # Load the dist row
    get_mat_from_blocks(block_directory, blockFlag, bi, distMat)
    # Since multiprocessing will pickle, we have to get return values
    # But also not send too much data
    tmpargs = []
    for ii in range(0, constants.BLOCK_SIZE):
        mi = constants.BLOCK_SIZE*bi+ii
        if mi<constants.N_NODE-1:
            tmpargs.append((distMat[ii,:], prevMat[ii,:], nextMat[ii,:], nodeFlag, hedInd[mi], hedVal[mi], mi))
    with multiprocessing.Pool(processes=nCores) as pool:
        results = pool.starmap(sort_ii, tmpargs)

    for ii in range(0, constants.BLOCK_SIZE):
#            sort_ii2(self.distMat, self.prevMat, self.nextMat, self.nodeFlag, self.hedInd, self.hedVal, bi, ii)
        mi = constants.BLOCK_SIZE*bi+ii
        if mi<constants.N_NODE-1:
            result = results[ii]
            prevMat[ii,], nextMat[ii,], hedInd[mi], hedVal[mi] = result

    distribute_mat_to_blocks(prevMat,blockFlag,bi,bprev)
    distribute_mat_to_blocks(nextMat,blockFlag,bi,bnext)
    # Only return subset that we calculated
    bil, bir = constants.BLOCK_SIZE*bi, constants.BLOCK_SIZE*(bi+1)
    if bir < len(hedInd):
        return bi, hedInd[bil:bir], hedVal[bil:bir]
    else:
        return bi, hedInd[bil:], hedVal[bil:]


def recalc_blocks(self, bk, ii, jj, subHedInd, subHedVal):
    self.prevMat = lManager.get_lprevMat()
    self.nextMat = lManager.get_lnextMat()
    self.distMat = lManager.get_ldistMat()
    self.beditPrev = self.lManager.get_lbeditPrev()
    self.beditNext = self.lManager.get_lbeditNext()

    [bii, iii] = constants.getbi(ii);
    [bjj, jjj] = constants.getbi(jj);

    # Update the hedInd, hedVal (cant assume we're getting the same one as before)
    bkl = bk * constants.BLOCK_SIZE
    bkr = (bk+1) * constants.BLOCK_SIZE
    if bkr < len(self.hedInd):
        self.hedInd[bkl:bkr] = subHedInd
        self.hedVal[bkl:bkr] = subHedVal
    else:
        self.hedInd[bkl:] = subHedInd
        self.hedVal[bkl:] = subHedVal            
    
    # bk < bii faster?
    if bk in range(0,bii):
        prepare_block_data(self.bdist,self.bprev,self.bnext,self.distMat,
                           self.prevMat,self.nextMat,self.beditPrev,
                           self.beditNext,self.blockFlag,bk)

        # Parallelize: moving to ctypes will be faster
        # Copying row costs especially 
        tmpargs = []
        for kk in range(0,constants.BLOCK_SIZE):
            mk = constants.getmi(bk,kk)
            if self.nodeFlag[mk]:
                self.hedInd[mk],self.hedVal[mk]=del2ins1(self.distMat[kk,:], self.prevMat[kk,:],
                                                         self.nextMat[kk,:], self.hedInd[mk], 
                                                         kk, ii, jj, self.beditPrev, self.beditNext)
                self.hedInd[mk],self.hedVal[mk]=del2ins1_local(self.hedInd[mk], kk, ii, jj)
#                    tmpargs.append((self.distMat[kk,:], self.prevMat[kk,:],
#                                                             self.nextMat[kk,:], self.hedInd[mk], 
#                                                             kk, ii, jj, self.beditPrev, self.beditNext))


        with multiprocessing.Pool(processes=self.nCores) as pool:
            results = pool.starmap(del2ins1, tmpargs)

        update_blocks(self.bprev, self.beditPrev, bk)
        update_blocks(self.bnext, self.beditNext, bk)    

    # bk == bii faster?
    elif bk in range(bii,bii+1):
        prepare_block_data(self.bdist,self.bprev,self.bnext,self.distMat,self.prevMat,
                           self.nextMat,self.beditPrev,self.beditNext,self.blockFlag,bk)

        # Parallelize
        for kk in range(0,iii):
            mk = constants.getmi(bk,kk)
            if self.nodeFlag[mk]:
                self.hedInd[mk],self.hedVal[mk]=del2ins1(self.distMat[kk,:], self.prevMat[kk,:],
                                                         self.nextMat[kk,:], self.hedInd[mk], 
                                                         kk, ii, jj, self.beditPrev, self.beditNext)

        # hand iith row
        self.nodeFlag[jj]=False
        self.hedInd[ii],self.hedVal[ii]=gen_pointers3(self.distMat[iii,:], self.nodeFlag,
                                                      ii, iii, self.beditPrev, self.beditNext)

        if bii==bjj:
            endRowInd=jjj
        else:
            endRowInd=constants.BLOCK_SIZE
        for kk in range(iii+1,endRowInd):
            mk = constants.getmi(bk,kk)
            if self.nodeFlag[mk]:
                self.hedInd[mk],self.hedVal[mk] = del_pointers(self.distMat[kk,:], self.prevMat[kk,:],
                                                               self.nextMat[kk,:], self.hedInd[mk],
                                                               kk, jj, self.beditPrev, self.beditNext)

        update_blocks_rowinsertion(self.bprev, self.beditPrev, bk)
        update_blocks_rowinsertion(self.bnext, self.beditNext, bk)

    elif bk in range(bii+1,bjj+1):
        prepare_block_data(self.bdist,self.bprev,self.bnext,self.distMat,self.prevMat,self.nextMat,
                           self.beditPrev,self.beditNext,self.blockFlag,bk)

        if bk==bjj:
            # jjj is the boundary; we hit the bottom row
            endRowInd=jjj
        else:
            endRowInd=constants.BLOCK_SIZE
        # Parallelize
        for kk in range(0,endRowInd):
            mk = constants.getmi(bk,kk)
            if self.nodeFlag[mk]:
                self.hedInd[mk],self.hedVal[mk] = del_pointers(self.distMat[kk,:], self.prevMat[kk,:],
                                                               self.nextMat[kk,:], self.hedInd[mk],
                                                               kk, jj, self.beditPrev, self.beditNext)

        update_blocks(self.bprev, self.beditPrev, bk)
        update_blocks(self.bnext, self.beditNext, bk)     

    if bkr < len(self.hedInd):
        return bk, self.hedInd[bkl:bkr], self.hedVal[bkl:bkr]
    else:
        return bk, self.hedInd[bkl:], self.hedVal[bkl:]

n, d = map(int, sys.argv[3:5])
constants.init(n, d) 
print("????")
print(constants.N_NODE)
block_directory, data_directory = sys.argv[1:3]
worker_id = sys.argv[5]
worker = CoreServer(n, d, block_directory, data_directory, worker_id)
print('worker %s exit.' % nodename)
