#!/usr/bin/env python

# models for which diagnostic plots are made
models = {
   '5Magb/': dict( mesa_profile='/hits/basement/pso/baumants/5M/relaxation-runs/ic/mesa-profile/mesa-profile-5Magb.data', ),
   '5Magb_0.0corotation_0.8rlof/': dict( mesa_profile='/hits/basement/pso/baumants/5M/relaxation-runs/ic/mesa-profile/mesa-profile-5Magb.data', ),
   '5Magb_1.0corotation_0.95rlof/': dict( mesa_profile='/hits/basement/pso/baumants/5M/relaxation-runs/ic/mesa-profile/mesa-profile-5Magb.data', ),
   #'name-of-model-2': dict( mesa_profile='ic/mesa-profile/sample-profile-deleted-parts.data', ),
}

quantites = ['rho', 'mach_number']
perspectives = ['xy', 'xz']

#plotting stuff (in units of the sun)
#for the slice plots
BOXSIZE = 400  #in Rsol
center_on_giant = True  # If set to false, center of mass will be the center

MIN_RHO = 1e-11  #in cgs
MAX_RHO = 1e-3   #in cgs

MIN_MACH = 0
MAX_MACH = 1

#for the radial plots
MAX_RADIUS = 1500
MIN_RADIUS = 1

# number of threads used for computations
number_of_cores = 10 # change this number to what you can afford
POOLSIZE = number_of_cores
NUMTHREADS = 2

# some constants and units
seconds_per_day = 86400.
seconds_per_hour = 3600.
