import numpy as np
from numpy import linalg as LA
from GBB import *
import matplotlib.pyplot as plt
import sys
import time 
from numpy import savetxt
from numpy import loadtxt


trials = int(loadtxt('trials.csv'))
T = int(loadtxt('T.csv'))
nb_run = int(loadtxt('nb_run.csv'))

tab_regret = [[[[] for i in range(nb_run)] for j in range(3)] for k in range(trials)]
for trial in range(trials) : 
	for c in range(3):
		for run in range(nb_run):
			tab_regret[trial][c][run] = loadtxt('tab_regret_'+str(trial)+'_'+str(c)+'_'+str(run)+'.csv')


M = []
m1 = []
m2 = []
m = []
gamma = []
alpha1 = []
epsilon = []
alpha2 = []
global_r_max = []

for trial in range(trials):
	M.append(loadtxt('M_'+str(trial)+'.csv'))
	m1.append(loadtxt('m1_'+str(trial)+'.csv'))
	m2.append(loadtxt('m2_'+str(trial)+'.csv'))
	m.append(loadtxt('m_'+str(trial)+'.csv'))
	gamma.append(loadtxt('gamma_'+str(trial)+'.csv'))
	alpha1.append(loadtxt('alpha1_'+str(trial)+'.csv'))
	epsilon.append(loadtxt('epsilon_'+str(trial)+'.csv'))
	alpha2.append(loadtxt('alpha2_'+str(trial)+'.csv'))
	global_r_max.append(loadtxt('global_r_max_'+str(trial)+'.csv'))

tab_regret_fraction = tab_regret
for i in range(trials):
	tab_regret_fraction[i] = tab_regret[i]/global_r_max[i]
	
mean_trial = np.mean(tab_regret_fraction, axis=0)
mean_run = np.mean(mean_trial, axis=1)

mean_algo1 = mean_run[0]
mean_algo2 = mean_run[1]
mean_ETC = mean_run[2]

plt.axhline(np.mean(alpha1), color='green', ls=':', linewidth=3, label=r"$\alpha1$")
plt.axhline(np.mean(alpha2), color='grey', ls='-.', linewidth=3, label=r"$\alpha2$")


window = 100
linewidth = 1.0
steps = 10
markersize = 7
plt.plot(np.arange(len(mean_algo1)), mean_algo1, color=(0, 0.4470, 0.7410, 0.1))

ma1 = np.convolve(mean_algo1, np.ones(window), 'valid')/window

plt.plot(np.arange(len(ma1)), ma1, marker="D", markevery=(T//steps), markersize=markersize, color=(0, 0.4470, 0.7410), linewidth=linewidth, label="OFUL for GBB")

plt.plot(np.arange(len(mean_algo2)), mean_algo2, color=(0.8500, 0.3250, 0.0980, 0.1))

ma2 = np.convolve(mean_algo2, np.ones(window), 'valid')/window

plt.plot(np.arange(len(ma2)), ma2, marker="v", markevery=(T//steps), markersize=markersize, color=(0.8500, 0.3250, 0.0980), linewidth=linewidth, label="Improved OFUL for GBB")

plt.plot(np.arange(len(mean_ETC)), mean_ETC, color=(0.4660, 0.6740, 0.1880, 0.1))

ma3 = np.convolve(mean_ETC, np.ones(window), 'valid')/window
plt.plot(np.arange(len(ma3)), ma3, marker="^", markevery=(T//steps), markersize=markersize, color=(0.4660, 0.6740, 0.1880), linewidth=linewidth, label="GBB-BAI")


plt.ylim(0.4,1)
plt.xlim(0,T)


plt.legend()
plt.grid()

plt.xlabel(r'$T$')
plt.ylabel("fraction of the optimal global reward")
plt.tight_layout()

plt.savefig("fig2.pdf", format="pdf", bbox_inches="tight")


