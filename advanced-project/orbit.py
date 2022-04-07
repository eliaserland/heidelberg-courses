#!/usr/bin/env python3
# coding: utf-8
import matplotlib.pyplot as plt
plt.ioff()
from arepo_run import Run 
import numpy as np
from const import *

pref='/hits/fast/pso/olofsses/5agb/ce-runs/'

def calc_orbits(path=''):
    """ Calculate the no. of orbits for the binary system as a function of time.
    """
    #print(pref+path+'/output')
    r = Run(pref+path+'/output')
    p = r.loadBinaryData()

    dd_orbits = {}        
    dd_orbits['time'] = p.time
    dd_orbits['n']    = np.zeros_like(p.time) # pre-allocation

    # Relative positions
    x_rel = p.poscomp[:, 0] - p.posrg[:,0]
    y_rel = p.poscomp[:, 1] - p.posrg[:,1]

    # Initial angle
    theta_init = np.arctan2(y_rel[0], x_rel[0])

    # Rotate coordinate system around z-axis to align rg & comp at x-axis.
    x_rel_new = x_rel * np.cos(theta_init) +  y_rel * np.sin(theta_init)     
    y_rel_new = -x_rel * np.sin(theta_init) +  y_rel * np.cos(theta_init)  

    theta = np.arctan2(y_rel_new, x_rel_new) # Relative angle
    if theta[1] < 0: # If clockwise rotation, reverse theta direction 
        theta = -theta
    theta[0] = 0 # Signbit can give problem if this is not done. 

    rev_fraction = theta.copy()
    rev_fraction[rev_fraction < 0] += np.pi
    rev_fraction /= 2*np.pi # Fractions of a revolution

    half_revs = np.zeros_like(p.time)
    half_revs[1:] = 0.5*np.cumsum(np.diff(np.signbit(theta))) # No. of completed half orbits

    dd_orbits['n'] = rev_fraction + half_revs 

    return dd_orbits

