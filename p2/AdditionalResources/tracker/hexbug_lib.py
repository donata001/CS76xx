import os 
import shutil
from math import *

OUTPUT_DIR = "output"
DEBUG_DIR = "debug"
INPUT_DIR = "input"

BIGINT = 999999999

# -----------------------------------------------
# math 

def movingAvg(a, count=3):
    ret = np.cumsum(a, dtype=float)
    ret[count:] = ret[count:] - ret[:-count]
    return np.around(ret[count - 1:] / count, decimals=3)

def movingAvg2D(a, count=3):
    a = np.array(a)
    ret = np.zeros(shape=(len(a)-(count-1), len(a[0])), dtype=np.float)

    for i in range(len(a[0])):
        vals = a[:,i]
        avg = movingAvg(vals)
        ret[:,i] = avg[:]
    return ret 

def distance(pa, pb):
    return sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)

def getAveragePoints(points):
    return np.sum(np.array(points), axis=0) / len(points)

def distanceList(pa, pb):
    pa, pb = np.array(pa), np.array(pb)
    diffx = (pa[:,0]-pb[:,0])**2 
    diffy = (pa[:,1]-pb[:,1])**2 
    return sum(np.sqrt(diffx + diffy))

# ----------------------------------------------- 
# perform basic file directory checks/cleanup 

def run_setup(resetDebug=False, resetOutput=False, checkInput=False):
    if checkInput and not os.path.exists(INPUT_DIR):
        raise OSError("Input directory [ ./"+INPUT_DIR+" ] not found!")

    # remake clean folders 
    if resetDebug and os.path.exists(DEBUG_DIR):
        shutil.rmtree(DEBUG_DIR)

    if not os.path.exists(DEBUG_DIR):
		os.makedirs(DEBUG_DIR)
    
    if resetOutput and os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    if not os.path.exists(OUTPUT_DIR):
	    os.makedirs(OUTPUT_DIR)

def read_data(datafile):
    positions = []
    for line in open(datafile, 'r'):
        pos = line.strip().split(',')
        positions.append([int(pos[0]), int(pos[1])])
    return positions