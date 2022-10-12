import numpy as np
import torch
import torch.nn as nn
from numpy import linalg as LA
from numpy.linalg import inv
from torch.optim import lr_scheduler
import torch.optim as optim
import itertools

import matplotlib.pyplot as plt
#Franck-Wolfe

def frank_wolfe(gbb):

    tmp = np.random.randint(0,100,gbb.K)
    tmp = tmp/np.sum(tmp)
    mu = nn.Parameter(torch.FloatTensor(tmp))

    print(mu)

    f = torch.FloatTensor([gbb.d+1])
    nb_iter = 1

    while torch.abs(f- gbb.d) > 0.001 :

        #Search for s:

        f, index_x = gbb.convexFX(mu)

        if nb_iter % 100 == 0 :
            print(f.item())

        f.backward()
        f_grad = mu.grad


        minimum = np.inf
        argmin = -1
        for idx in range(gbb.K):
            s = torch.zeros(gbb.K)
            s[idx] = 1

            value = torch.mm(s.view(1,-1), f_grad.view(-1,1))

            if value < minimum : 
                minimum = value
                argmin = s

        step_size = 2/(2+nb_iter)

        mu = nn.Parameter(mu.data + step_size*(argmin - mu.data))

        nb_iter = nb_iter + 1

        if nb_iter == 20000:
            break


    mu = mu.data
    return mu