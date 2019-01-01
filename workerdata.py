import numpy as np
from multiprocessing import cpu_count
from multiprocessing.managers import BaseManager, NamespaceProxy
from blockfilemmap import BlockFileMap
import common_base as cb

class LocalManager(BaseManager):
    pass

class ArrayProxy(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__', '__setitem__', '__getitem__')
    def __getitem__(self, item):
        return self._callmethod('__getitem__', (item,))
    def __setitem__(self, item, val):
        self._callmethod('__setitem__', (item,val,))

class EditPoolProxy(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__', 'clear',
                 'insert_edit_rep', 'insert_row_edit', 'sort_edit')
    def clear(self, bi):
        self._callmethod('clear', (bi,))
    def insert_edit_rep(self, rowind, ind, val):
        self._callmethod('insert_edit_rep', (rowind,ind,val,))
    def insert_row_edit(self, rowind, ind, val):
        self._callmethod('insert_row_edit', (rowind,ind,val,))
    def sort_edit(self, bi):
        self._callmethod('sort_edit', (bi,))

lManager = LocalManager()

class WorkerData():

    # Worker data
    # Only one set of this data to exist per node
    beditPrev = editPool()
    beditNext = editPool()
    LocalManager.register('get_lbeditPrev', proxytype=EditPoolProxy, exposed=None, callable=lambda: beditPrev)
    LocalManager.register('get_lbeditNext', proxytype=EditPoolProxy, exposed=None, callable=lambda: beditNext)
    nCores = cpu_count()

    # Globals required for manager to work without returns
    hedInd = None
    hedVal = None
    prevMat = None
    nextMat = None
    distMat = None
    bprev = None
    bnext = None
    bdist = None    
    nodeFlag = None
    blockFlag = None
    blockCount = None

    @classmethod
    def init(self, block_directory, base_directory):
        # Shared globals
        n = constants.N_NODE
        print(n)
        hedInd = np.zeros(n-1,dtype=constants.DATA_TYPE)
        hedVal = np.zeros(n-1,dtype=constants.DATA_TYPE)

        # Set up constants and local variables
        bs, nb = constants.BLOCK_SIZE, constants.N_BLOCK
        prevMat = np.zeros((bs,nb*bs),dtype=constants.DATA_TYPE)
        nextMat = np.zeros((bs,nb*bs),dtype=constants.DATA_TYPE)
        distMat = np.zeros((bs,nb*bs),dtype=constants.DATA_TYPE)
        LocalManager.register('get_lprevMat', proxytype=ArrayProxy, exposed=None, callable=lambda: prevMat)
        LocalManager.register('get_lnextMat', proxytype=ArrayProxy, exposed=None, callable=lambda: nextMat)
        LocalManager.register('get_ldistMat', proxytype=ArrayProxy, exposed=None, callable=lambda: distMat)
        lManager.start()

        bprev = np.zeros((constants.N_BLOCK,constants.N_BLOCK),dtype=object)
        bnext = np.zeros((constants.N_BLOCK,constants.N_BLOCK),dtype=object)
        bdist = np.zeros((constants.N_BLOCK,constants.N_BLOCK),dtype=object)

        # ***** DO DYNAMICALLY? WASTE OF TIME?
        # Create references to blocks - may be better to do dynamically
        for bi in range(len(self.bprev)):
            for bj in range(bi, len(self.bprev)):
                bfn = block_directory+"/{}_n/{}_n.block".format(bi, bj)
                bfp = block_directory+"/{}_p/{}_p.block".format(bi, bj)
                bfd = block_directory+"/{}_d/{}_d.block".format(bi, bj)
                shape = (constants.BLOCK_SIZE, constants.BLOCK_SIZE)
                bdist[bi, bj] = BlockFileMap(bfd, constants.DATA_TYPE, shape)
                bnext[bi, bj] = BlockFileMap(bfn, constants.DATA_TYPE, shape)
                bprev[bi, bj] = BlockFileMap(bfp, constants.DATA_TYPE, shape)
                
        # Flags
        nodeFlag = np.zeros(constants.N_BLOCK*constants.BLOCK_SIZE, dtype=bool)
        nodeFlag[:n]=True
        blockFlag = np.ones(constants.N_BLOCK)>0
        blockCount = np.zeros(constants.N_BLOCK)+constants.BLOCK_SIZE
        if constants.N_NODE%constants.BLOCK_SIZE!=0:
            blockCount[-1]=constants.N_NODE%constants.BLOCK_SIZE

        # Get number of available cores

    @classmethod
    def update_nodeflag(self, jj):
        print("updating nodeflag", jj)
        nodeFlag[jj]=False
        if jj<constants.N_NODE-1:
            hedInd[jj]=constants.DEL_VAL
            hedVal[jj]=constants.DEL_VAL

    @classmethod
    def update_blockflag(self, bjj):
        print("updating blockflag", bjj)
        blockCount[bjj] -= 1
        blockFlag[bjj] = self.blockCount[bjj]>0

