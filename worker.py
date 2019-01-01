import time, sys, socket, random
import numpy as np
from multiprocessing import Pool
from linkage_functions import *
import multiprocessing
from multiprocessing import cpu_count
from multiprocessing.managers import BaseManager, NamespaceProxy
from update_map import UpdateMap
from blockfilemmap import BlockFileMap

class Worker():

    @classmethod

