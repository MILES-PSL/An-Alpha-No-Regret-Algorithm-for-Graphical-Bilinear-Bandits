import numpy as np
from numpy import linalg as LA
from GBB import *
import matplotlib.pyplot as plt
import sys
import time 
from numpy import savetxt


np.random.seed(8)

n = 5 # Number of agents

d = 10 # Dimension of an arm x in X 

K = 10 # Number of arms in X

L = 1 # Norm of an arm in X

T = 2000 # Horizon

nb_run = 1 #Number of runs to plot the expected-regret

S = 10 # Norm of the unknown parameter theta

sigma = 1 # Variance of the noise in the reward

delta = 0.001 # Confidence parameter for the upper bound

trials = 100 # Number of trials we do to test the algorithm with a different matrix M and arm set each time

tab_gamma =[[] for i in range(trials)]
tab_epsilon =[[] for i in range(trials)]
tab_alpha1 = [[] for i in range(trials)]
tab_alpha2 = [[] for i in range(trials)]

step = 100
for trial in range(trials):

    print("Trial n°%d/%d" % (trial+1, trials))

    # Generate matrix M, and keep it if the best reward is not in the diagonal (otherwise the best allocation is trivial).
    while True : 

        X = np.eye(d)

        M_star = np.abs(np.random.randn(d,d))
        M_star = (M_star/LA.norm(M_star))*S

        typeOfGraph = "complete"

        gbb = GBB(typeOfGraph=typeOfGraph, n=n, X=X, M=M_star, sigma=sigma)
        (x_max , x_prime_max), r_max, (x_min , x_prime_min), r_min = gbb.argmaxminBilinearReward()
        print(x_max, x_prime_max, r_max)
        if x_max != x_prime_max : 
            break

    M_tilde = gbb.M

    allocation_max, global_r_max, allocation_min, global_r_min = gbb.argmaxminReward()

    print("Set of nodes : "+str(gbb.V))
    print("Set of edges : "+str(gbb.E))

    print("Best edge arm : (%d,%d) - reward = %.2f" % (x_max, x_prime_max, r_max))
    print("Worst edge arm : (%d,%d) - reward = %.2f" % (x_min, x_prime_min, r_min))

    print("Best allocation : "+ str(allocation_max) +" - reward = %.2f" % (global_r_max))
    print("Worst allocation : "+ str(allocation_min) +" - reward = %.2f" % (global_r_min))
    
    zeta = 1/step
    
    for i in range(step+1):
    
        M_tilde[x_max,x_max]=(i*zeta)*r_max/2
        M_tilde[x_prime_max, x_prime_max]=(i*zeta)*r_max/2

        gbb.M = M_tilde

        m12 = gbb.m12
        m21 = gbb.m21
        m1 = gbb.m1
        m2 = gbb.m2

        gamma = np.min([M_tilde[j,j] for j in range(K)])/(global_r_max/gbb.m)
        alpha1 = (1+gamma)/2
        r1= m12*M_tilde[x_max,x_prime_max] + m21*M_tilde[x_prime_max,x_max] +m1*M_tilde[x_max,x_max]+m2*M_tilde[x_prime_max,x_prime_max]
        r2= np.max([m12*M_tilde[i,j] +m21*M_tilde[j,i] + m1*M_tilde[i,i] + m2*M_tilde[j,j] for i in range(K) for j in range(K)])
        delta = r2-r1
        epsilon = delta/global_r_max
        alpha2 = 1-(((m1+m2)/gbb.m)*(1-gamma)-epsilon)
        tab_gamma[trial].append(gamma)
        tab_epsilon[trial].append(epsilon)
        tab_alpha1[trial].append(alpha1)
        tab_alpha2[trial].append(alpha2)


# plt.plot(np.arange(step+1)/step, np.mean(tab_gamma, axis=0), color="C0", label="gamma")
# plt.plot(np.arange(step+1)/step, np.mean(tab_alpha1, axis=0), color="C1", label = "alpha 1")
# plt.plot(np.arange(step+1)/step, np.mean(tab_epsilon, axis=0), color="C2", label = "epsilon")
# plt.plot(np.arange(step+1)/step, np.mean(tab_alpha2, axis=0), color="C3", label = "alpha 2")
# plt.legend()

# plt.savefig("expe1.png")


savetxt('gamma.csv', np.mean(tab_gamma, axis=0), delimiter=',')
savetxt('alpha1.csv', np.mean(tab_alpha1, axis=0), delimiter=',')
savetxt('epsilon.csv', np.mean(tab_epsilon, axis=0), delimiter=',')
savetxt('alpha2.csv', np.mean(tab_alpha2, axis=0), delimiter=',')

# to generate the associated figure, use load_fig1.py
