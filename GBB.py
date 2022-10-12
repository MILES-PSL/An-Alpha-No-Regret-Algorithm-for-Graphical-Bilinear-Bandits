import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import itertools
import sys
from functools import reduce
import torch
import torch.nn as nn

import matplotlib.pyplot as plt


class GBB:
    def __init__(self, typeOfGraph="complete", n=2, X=np.eye(2), M=np.eye(2), sigma=1):

        self.typeOfGraph = typeOfGraph
        self.n = n
        self.X = X
        self.K, self.d = self.X.shape
        self.d2 = self.d**2
        self.Z = self.fromXtoZ()

        self.M = M
        self.MX = self.X @ self.M @ self.X.T
        self.sigma = sigma

        self.V = np.arange(n)
        self.E = self.createEdgeSet()
        self.N = self.createNeighboursSets()
        self.V1, self.V2 = self.approx_max_cut()
        self.m = len(self.E)
        self.m12=0
        self.m21=0
        self.m1=0
        self.m2=0
        for i in self.V1:
            for j in self.N[i]:
                if j in self.V2:
                    self.m12=self.m12+1
                elif j in self.V1:
                    self.m1 = self.m1+1
        for i in self.V2:
            for j in self.N[i]:
                if j in self.V1:
                    self.m21=self.m21+1
                elif j in self.V2:
                    self.m2 = self.m2+1

        self.m = len(self.E)
        self.Z1 = self.fromZtoZ1()
        self.Z2 = self.fromZtoZ2()

    def vec(self, M):
        return (M.T).reshape(-1,1)

    def fromXtoZ(self):
        Z = np.zeros((self.K**2, self.d**2))
        for i in range(self.K):
            for j in range(self.K):
                Z[i * self.K + j] = self.vec(np.outer(self.X[i], self.X[j])).reshape(-1)
        return Z

    def createEdgeSet(self):
        if self.typeOfGraph == "circle" :
            return [(i, i+1 if i+1 < self.n else 0) for i in range(self.n)] + [(i+1 if i+1 < self.n else 0, i) for i in range(self.n)]
        elif self.typeOfGraph == "star":
            return [(0, i) for i in range(1,self.n)] + [(i, 0) for i in range(1,self.n)]
        elif self.typeOfGraph == "root":
            return [(i,i+1) for i in range(self.n-1)] + [(i+1, i) for i in range(self.n-1)]
        elif self.typeOfGraph == "complete":
            E = []
            for pair in itertools.combinations(self.V, 2):
                E.append((pair[0],pair[1]))
                E.append((pair[1],pair[0]))
            return E
        elif self.typeOfGraph == "random":
            E = []
            for pair in itertools.combinations(self.V, 2):
                bool  = np.random.randint(0,10, 1)
                if bool > 3 and pair[0] != pair[1]:
                    E.append((pair[0],pair[1]))
                    E.append((pair[1],pair[0]))
            return E
        elif self.typeOfGraph == "matching":
            return [ (int(i), int(i+1)) for i in np.arange(self.n//2)*2] + [ (int(i+1), int(i)) for i in np.arange(self.n//2)*2]
        else : 
            print("Error : ", typeOfGraph, " grpah is not supported")
            sys.exit(-1)
    def createNeighboursSets(self):
        N = [[] for i in range(self.n)]
        for i in self.V :
            for (j,k) in self.E :
                if i==j:
                    N[i].append(k)
        return N

    def getReward(self, allocation, i=None, noise=True):
        if noise == True :
            noise = 1
        else : 
            noise = 0

        if i is None :
            return [((allocation[j], allocation[k]), self.MX[allocation[j], allocation[k]] + noise * np.random.normal(0,self.sigma,1)[0]) for (j,k) in self.E]
        else : 
            return [((allocation[i], allocation[j]), self.MX[allocation[i], allocation[j]] + noise * np.random.normal(0,self.sigma,1)[0]) for j in self.N[i]]

    def getRewardBis(self, allocation, noise=True):
        if noise == True :
            noise = 1
        else : 
            noise = 0

        Z = np.array([self.Z[allocation[i]*self.K + allocation[j]] for (i,j) in self.E])
        y = np.array([self.MX[allocation[j], allocation[k]] + noise * np.random.normal(0,self.sigma,1)[0] for (j,k) in self.E]).reshape(-1,1)
        
        return y, Z
    def getExpectedGlobalReward(self, allocation):
        return np.sum([self.MX[allocation[j], allocation[k]] for (j,k) in self.E])

    def argmaxminReward(self):
        allocation = [0 for i in range(self.n)]
        allocations = []
        i = 0
        while True :
            allocations.append(allocation.copy())
            if (allocation == np.ones(self.n)*(self.K-1)).all() :
                break
            index = len(allocation)-1
            
            while index>=0 : 
                if allocation[index]==self.K-1:
                    allocation[index] = 0
                    index = index -1
                else : 
                    allocation[index] = allocation[index] +1
                    break
            i = i + 1

        max_ = - np.inf
        argmax = -1
        min_ = np.inf
        argmin = -1
        for allocation in allocations:
            r = self.getExpectedGlobalReward(allocation)
            if r > max_:
                max_ = r
                argmax = allocation
            if r < min_:
                min_ = r
                argmin = allocation

        return [argmax, max_, argmin, min_]

    def argmaxminBilinearReward(self):
        M = self.MX + self.MX.T
        argmax = np.argmax(M)
        i, j = (argmax//self.K), argmax%self.K
        max_ = M[i,j]
        argmin = np.argmin(M)
        k, l = (argmin//self.K), argmin%self.K
        min_ = M[k,l]

        return [(i,j), max_, (k,l), min_]
    def approx_max_cut(self):
        V1 = []
        V2 = []
        for i in self.V:
            n1 = 0
            n2 = 0
            for j in self.N[i]:
                if j in V1:
                    n1= n1 + 1
                elif j in V2 : 
                    n2= n2 + 1
            if n1>n2 : 
                V2.append(i)
            else : 
                V1.append(i)
        return (V1, V2)
    def fromZtoZ1(self):
        Z_prime = np.zeros_like(self.Z)
        for i in range(self.K):
            for j in range(i, self.K):
                Z_prime[i*self.K+j]=self.Z[i*self.K+j] + self.Z[j*self.K+i]
                Z_prime[j*self.K+i]=self.Z[i*self.K+j] + self.Z[j*self.K+i]
        return Z_prime

    def fromZtoZ2(self):
        Z_prime = np.zeros_like(self.Z)
        for i in range(self.K):
            for j in range(i, self.K):
                Z_prime[i*self.K+j]=self.m12*self.Z[i*self.K+j] + self.m21*self.Z[j*self.K+i] + self.m1*self.Z[i*self.K+i] + self.m2*self.Z[j*self.K+j] 
                Z_prime[j*self.K+i]=self.m12*self.Z[j*self.K+i] + self.m21*self.Z[i*self.K+j] + self.m1*self.Z[j*self.K+j] + self.m2*self.Z[i*self.K+i] 
        return Z_prime
    
    def convexFX(self, mu, nbOfDraw=1):

        A = 0
        for a in range(self.K):
            A = A + mu[a] * torch.mm(torch.from_numpy(self.X[a]).float().view(-1,1), torch.from_numpy(self.X[a]).float().view(1,-1))

        try:
            invA = torch.inverse(A)
        except:
            invA = torch.inverse(A+torch.eye(len(A)))

        maximum = -np.inf
        argmax = -1

        for i, x in enumerate(self.X) : 
            value = torch.mm(torch.mm(torch.from_numpy(x).float().view(1,-1), invA), torch.from_numpy(x).float().view(-1,1))
                
            if value > maximum : 
                maximum = value
                argmax = i

        return maximum, argmax





