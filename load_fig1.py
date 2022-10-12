import numpy as np
from numpy import linalg as LA
from GBB import *
import matplotlib.pyplot as plt
import sys
import time 
from numpy import savetxt
from numpy import loadtxt


gamma = loadtxt('gamma.csv', delimiter=',')
alpha1 = loadtxt('alpha1.csv', delimiter=',')
epsilon = loadtxt('epsilon.csv', delimiter=',')
alpha2 = loadtxt('alpha2.csv', delimiter=',')

linewidth = 3.0
plt.plot(np.arange(len(gamma))/(len(gamma)-1), gamma, color="C0", label=r'$\xi$', ls="-", linewidth=linewidth)
plt.plot(np.arange(len(alpha1))/(len(alpha1)-1), alpha1, color="C1", label=r'$\alpha_1$', ls="--", linewidth=linewidth)
plt.plot(np.arange(len(epsilon))/(len(epsilon)-1), epsilon, color="C2", label=r'$\epsilon$', ls="-.", linewidth=linewidth)
plt.plot(np.arange(len(alpha2))/(len(alpha2)-1), alpha2, color="C3", label=r'$\alpha_2$', ls=":", linewidth=linewidth)
plt.legend()
plt.grid()
plt.xlim(0,1)
plt.ylim(0,1)

plt.xlabel(r'$\zeta$')
plt.yticks(np.arange(0, 1.1, 0.1))

plt.savefig("fig1.pdf", format="pdf", bbox_inches="tight")


