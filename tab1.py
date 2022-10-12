import numpy as np
from numpy import linalg as LA
from GBB import *
import matplotlib.pyplot as plt
import sys
import time 
from numpy import savetxt


np.random.seed(8)

n = 100 # Number of agents

d = 10 # Dimension of an arm x in X 

S = 10 # Norm of the unknown parameter theta

sigma = 1 # Variance of the noise in the reward

trials = 100 # Number of trials we do to test the algorithm with a different matrix M and arm set each time

tab_m12 =[[] for i in range(trials)]
tab_m21 =[[] for i in range(trials)]
tab_m1 = [[] for i in range(trials)]
tab_m2 = [[] for i in range(trials)]

for trial in range(trials):

    print("Trial n°%d/%d" % (trial+1, trials))

    X = np.eye(d)

    M_star = np.abs(np.random.randn(d,d))
    M_star = (M_star/LA.norm(M_star))*S

    typeOfGraphs =["complete", "random", "circle", "star", "matching"]
    for typeOfGraph in typeOfGraphs:

        gbb = GBB(typeOfGraph=typeOfGraph, n=n, X=X, M=M_star, sigma=sigma)
        tab_m12[trial].append(gbb.m12/gbb.m)
        tab_m21[trial].append(gbb.m21/gbb.m)
        tab_m1[trial].append(gbb.m1/gbb.m)
        tab_m2[trial].append(gbb.m2/gbb.m)

print(typeOfGraphs)
print(np.mean(tab_m1, axis=0)+np.mean(tab_m2, axis=0))


