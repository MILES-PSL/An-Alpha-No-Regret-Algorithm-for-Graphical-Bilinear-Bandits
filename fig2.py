import numpy as np
from numpy import linalg as LA
from GBB import *
import matplotlib.pyplot as plt
import sys
import time 
from numpy import savetxt
from Frank_Wolfe import *
import torch
import torch.nn as nn


np.random.seed(2)


def OFUL(Z_prime, theta, A, invA, d, lbda, delta, S):
    t0= time.time()
    estimated_reward = Z_prime @ theta
    # t1 = time.time()-t0
    # print(t1)
    # t1 = time.time()
    # estimated_reward = Z @ theta
    # beta_bis = sigma * np.sqrt( 2 * np.log( (np.sqrt(LA.det(A)) ) / (np.sqrt(LA.det(np.eye(d)*lbda))*delta)  )  ) + np.sqrt(lbda) * S
    beta = sigma * np.sqrt((d**2)*np.log((1+t*(L**2)/lbda)/delta))+ np.sqrt(lbda) * S
    # t2 = time.time()-t1
    # print(t2)
    # t2 = time.time()
    optimistic_reward = estimated_reward + beta * np.sqrt((Z_prime @ invA @ Z_prime.T).diagonal()).reshape(-1,1)
    # t3 = time.time()-t2
    # print(t3)
    # t3 = time.time()
    # argmax = np.argmax(optimistic_reward)
    argmax = np.random.choice(np.flatnonzero(optimistic_reward == optimistic_reward.max()))
    # t4 = time.time()-t0
    # print(t4)
    return argmax

n = 5 # Number of agents

d = 5 # Dimension of an arm x in X 

K = d # Number of arms in X

L = 1 # Norm of an arm in X

T = 20000 # Horizon

nb_run = 10 # Number of runs to plot the expected-regret

S = 4 # Norm of the unknown parameter theta

lbda = 1 # Ridge regularization parameter

sigma = 1 # Variance of the noise in the reward

delta = 0.001 # Confidence parameter for the upper bound

trials = 5 # Number of trials we do to test the algorithm with a different matrix M and arm set each time

savetxt('trials.csv', [trials], delimiter=',')
savetxt('T.csv', [T], delimiter=',')
savetxt('nb_run.csv', [nb_run], delimiter=',')

tab_M = []
tab_m1 = []
tab_m2 = []
tab_m = []
tab_gamma = []
tab_alpha1 = []
tab_epsilon = []
tab_alpha2 = []
tab_global_r_max = []

tab_regret = [[[[] for i in range(nb_run)] for j in range(3)] for k in range(trials)]

for trial in range(trials):

    print("Trial n°%d/%d" % (trial+1, trials))
    while True : 

        X = np.eye(d)

        M_star = np.abs(np.random.randn(d,d))
        M_star = (M_star/LA.norm(M_star))*S


        typeOfGraph = "complete"

        gbb = GBB(typeOfGraph=typeOfGraph, n=n, X=X, M=M_star, sigma=sigma)

        (x_max , x_prime_max), r_max, (x_min , x_prime_min), r_min = gbb.argmaxminBilinearReward()
        gbb.M[x_max, x_max] = 0
        gbb.M[x_prime_max, x_prime_max] = 0

        allocation_max, global_r_max, allocation_min, global_r_min = gbb.argmaxminReward()

        if x_max != x_prime_max  and np.logical_or((allocation_max == x_max), (allocation_max==x_prime_max)).all()==False: 
            break
    
    m12 = gbb.m12
    m21 = gbb.m21
    m1 = gbb.m1
    m2 = gbb.m2
    gamma = np.min([gbb.M[j,j] for j in range(K)])/(global_r_max/gbb.m)
    alpha1 = (1+gamma)/2
    r1= m12*gbb.M[x_max,x_prime_max] + m21*gbb.M[x_prime_max,x_max] +m1*gbb.M[x_max,x_max]+m2*gbb.M[x_prime_max,x_prime_max]
    r2= np.max([m12*gbb.M[i,j] +m21*gbb.M[j,i] + m1*gbb.M[i,i] + m2*gbb.M[j,j] for i in range(K) for j in range(K)])
    dlta = r2-r1
    epsilon = dlta/global_r_max
    alpha2 = 1-(((m1+m2)/gbb.m)*(1-gamma)-epsilon)

    tab_M.append(gbb.M)
    tab_m1.append(m1)
    tab_m2.append(m2)
    tab_m.append(gbb.m)
    tab_gamma.append(gamma)
    tab_alpha1.append(alpha1)
    tab_epsilon.append(epsilon)
    tab_alpha2.append(alpha2)
    tab_global_r_max.append(global_r_max)

    savetxt('M_'+str(trial)+'.csv', gbb.M, delimiter=',')
    savetxt('m1_'+str(trial)+'.csv', [m1], delimiter=',')
    savetxt('m2_'+str(trial)+'.csv', [m2], delimiter=',')
    savetxt('m_'+str(trial)+'.csv', [gbb.m], delimiter=',')
    savetxt('gamma_'+str(trial)+'.csv', [gamma], delimiter=',')
    savetxt('alpha1_'+str(trial)+'.csv', [alpha1], delimiter=',')
    savetxt('epsilon_'+str(trial)+'.csv', [epsilon], delimiter=',')
    savetxt('alpha2_'+str(trial)+'.csv', [alpha2], delimiter=',')
    savetxt('global_r_max_'+str(trial)+'.csv', [global_r_max], delimiter=',')
    

    print("Set of nodes : "+str(gbb.V))
    print("Set of edges : "+str(gbb.E))

    print("Best edge arm : (%d,%d) - reward = %.2f" % (x_max, x_prime_max, r_max))
    print("Worst edge arm : (%d,%d) - reward = %.2f" % (x_min, x_prime_min, r_min))
    
    print("Best allocation : "+ str(allocation_max) +" - reward = %.2f" % (global_r_max))
    print("Worst allocation : "+ str(allocation_min) +" - reward = %.2f" % (global_r_min))
    print(gbb.M)
    # ##################################################################
    #   Find the G-Allocation distributon mu for explore then commit   #
    # ##################################################################
    
    mu = frank_wolfe(gbb)

    mu = mu.numpy()
    mu = mu/np.sum(mu)


    for c, Zi in enumerate([gbb.Z1, gbb.Z2, -1]):
        # if c<2:
        #     continue

        for run in range(nb_run):
            explore = True
            x_opt_exploit, x_prime_opt_eploit = -1, -1

            print("Run n°%d/%d" % (run+1, nb_run))
            # ###############################################
            # # Parameters                                  #
            # ###############################################
            A = np.eye(d**2) * lbda
            b = np.zeros((d**2,1))
            invA = np.eye(d**2) / lbda
            theta = np.zeros((d**2,1))
            Regret = 0

            for t in range(T):
                if c == 2 and t > T/3:
                    explore=False
                ##############################################
                # Learning Precessus                         #
                ##############################################

                # Choose an action
                # M = mat(theta)
                # print("t4 :", time.time())
                if c != 2:
                    i_z = OFUL(Zi, theta, A, invA, d**2, lbda, delta, S*2)
                    x, x_prime = int((i_z - (i_z % K))//K), (i_z % K)
                    allocation = [x_prime for i in range(n)]
                    for i in range(len(gbb.V1)):
                        allocation[i] = x
                else : 
                    if explore == True :
                        allocation = np.random.choice(gbb.K, size=n, p=mu)
                    else : 
                        if x_opt_exploit ==-1:
                            i_z = np.argmax(gbb.Z@theta)
                            x_opt_exploit, x_prime_opt_exploit = int((i_z - (i_z % K))//K), (i_z % K)
                        allocation = [x_prime_opt_exploit for i in range(n)]
                        for i in range(len(gbb.V1)):
                            allocation[i] = x_opt_exploit


                # Receive the reward
                y, Z_ = gbb.getRewardBis(allocation, noise= True)
                

                if c!=2 or explore == True:
                    A = A + Z_.T@Z_
                    b = b + Z_.T@y
                    invA = np.linalg.solve(A, np.eye(d**2))
                    theta = invA @ b
                global_reward = gbb.getExpectedGlobalReward(allocation)

                if t%1000==0:
                    print("Bras joués : "+ str(allocation) + " - reward : "+str(global_reward))

                # Update Regret
                Regret = global_reward
                tab_regret[trial][c][run].append(global_reward)

            savetxt('tab_regret_'+str(trial)+'_'+str(c)+'_'+str(run)+'.csv', tab_regret[trial][c][run], delimiter=',')


# To plot the results, use plotfig.py