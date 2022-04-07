import numpy as np
from arepo_run import Run
import matplotlib.pyplot as plt
plt.ioff()  # turn off plotting for other backends then saving
from const import *

########################################################################
 
# use Run class from arepo_run to load the data
#simulation_path = '/hits/fast/pso/olofsses/5agb/ce-runs/5Magb/output'
simulation_path = '/hits/fast/pso/olofsses/5agb/ce-runs/5Magb_1.0corotation_0.95rlof/output'
#simulation_path = '/hits/fast/pso/olofsses/5agb/ce-runs/5Magb_0.0corotation_0.8rlof/output'

#simulation_path = '/hits/basement/pso/baumants/1.5M/ce-runs/1.5Mrgb_q0.50/output'
plot_path = simulation_path + '/plots/other' # '.'
 
run = Run(simulation_path)

# load the simulation data from here
binary_file = run.loadBinaryData()
energy_file = run.loadEnergyData()

# Fix bug with inf pot energy of tracer particles, credit to Thomas
energy_file.epot = energy_file.epots[0] + energy_file.epots[1]               
energy_file.etot = energy_file.epot + energy_file.ein + energy_file.ekin

# make a figure of data from the txt files. Best look in the file gadget.py from line 887 in the snap utils to find out what the recorded quantities are
fig, ax = plt.subplots(1,1)
 
ax.plot(binary_file.time/86400, binary_file.distance/rsol)
ax.set_xlabel('Time / d')
ax.set_ylabel(r'Distance / $R_\odot$')
 
ax2 = ax.twinx()
ax2.plot(energy_file.time/86400, energy_file.etot , color = 'black')
#ax2.plot(energy_file.time/86400, energy_file.ekin) #, color = 'black')
ax2.set_ylabel(r'Energy / ergs')

#fig.tight_layout()
fig.savefig('{}/{}'.format(plot_path, 'distance.pdf'), bbox_inches="tight")

#----
#----
#----

fig, ax = plt.subplots(2,1, figsize=(8,5)) #, sharex=True)
ax = ax.flatten()

# First subplot
p1 = ax[0].semilogy(binary_file.time/86400, binary_file.distance/rsol, label='separation')
ax0_twin = ax[0].twinx()
p2 = ax0_twin.plot(energy_file.time/86400, (energy_file.etot-energy_file.etot[0])/energy_file.epot[0], label='rel. energy error', color='black')

ax[0].set_xlabel('Time / d')
ax[0].set_ylabel(r'Distance / $R_\odot$')
ax0_twin.set_ylabel(r'$(E_{tot}-E_{tot}[0])/E_{pot}[0]$')

lns = p1+p2
labs = [l.get_label() for l in lns]
ax[0].legend(lns, labs, loc=0)

# Second subplot
ax[1].plot(energy_file.time/86400, energy_file.etot, label='$E_{tot}$', color = 'black')
ax[1].plot(energy_file.time/86400, energy_file.ein, label='$E_{in}$')
ax[1].plot(energy_file.time/86400, energy_file.epot, label='$E_{pot}$')
ax[1].plot(energy_file.time/86400, energy_file.ekin, label='$E_{kin}$')

ax[1].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
ax[1].set_xlabel('Time / d')
ax[1].set_ylabel(r'Energy / ergs')

fig.tight_layout()
fig.savefig('{}/{}'.format(plot_path, 'distance2.pdf'), bbox_inches="tight")




########################################################################
# load a single snapshot like this:
s = run.loadSnap(0) # loads snap 0.. 
snap_cnt = run.snapcount
print(run.snapcount)

# the snapshots contain data with arrays that have an entry for each particle in the simulation
# access seperate particle types using the type array and masks
print('Resolution: {:.2e} particles , containing {:.2e} gas particles, {:.2e} tracers, {} dm particles'.\
        format(len(s.type), len(s.type[s.type==0]), len(s.type[s.type==2]), len(s.type[s.type==1])))

# s.data[]
# s.type 0 1 2
# dont use angualar momentum from binary file.

# check the mass distribution
mass_gas = s.mass[s.type==0].sum() # = envolope mass
mass_primary = s.mass[s.id==1e9][0] # the cores always have ids 1e9 and 1e9+1
mass_companion = s.mass[s.id==1e9+1][0]
print('Mass: {:.2f} in gas, {:.2f} in the primary core, {:.2f} in the companion core (in msun)'.\
    format(mass_gas/msol, mass_primary/msol, mass_companion/msol))

# other quantities include vel, acce, time, soft, ...

########################################################################
"""
# Temperature vs. radial distance to primary for all gas particles.
for snap_nr in range(run.snapcount):
	s = run.loadSnap(snap_nr) 
	pos_primary = s.pos[s.id==1e9]
	radius_gas = np.sqrt(((s.pos[s.type==0] - pos_primary)**2).sum(axis=1))

	plt.figure()
	plt.loglog(radius_gas/rsol, s.temp, ',')
	plt.xlabel(r'Radius / $R_\odot$')
	plt.ylabel(r'Temperature / K')
	plt.title(f'Snapshot #{snap_nr}')
	plt.savefig(f'{plot_path}/temp_radius_{snap_nr}.png', bbox_inches="tight", dpi=300)
	#plt.savefig('{}/{}'.format(plot_path, 'temp_radius.png'), bbox_inches="tight")
	plt.close('all')
"""
########################################################################
# the Run class has a powerful cache system that lets you easily compute stuff accross all snapshots
run.aggregateValues(['munbound', 'time'])

print('Simulation produced {} snapshots, simulation time at {:.2f}d'.format(run.snapcount, run.time[-1]/86400))
#-----------------------------------------

#print(run.time/86400)
fig, ax = plt.subplots(1,1)
 
ax.semilogy(binary_file.time/86400, binary_file.distance/rsol)
ax.set_xlabel('Time / d')
ax.set_ylabel(r'Distance / $R_\odot$')
 
ax2 = ax.twinx()
ax2.plot(energy_file.time/86400, energy_file.etot , color = 'black')
#ax2.plot(energy_file.time/86400, energy_file.ekin) #, color = 'black')
ax2.set_ylabel(r'Energy / ergs')

#print(len(run.time), run.time.shape)
#print(len(energy_file.etot), energy_file.etot.shape)
#print(len(energy_file.time), energy_file.time.shape)
#print(np.argmin((run.time-energy_file.time[:, None])**2, axis=0))

# Time visualization for snapshots.
ax2.plot(run.time/86400, energy_file.etot[np.argmin((run.time-energy_file.time[:, None])**2, axis=0)], '.', label='snapshots', color='red')

ax2.legend(loc='upper center')
fig.savefig('{}/{}'.format(plot_path, 'distance3.pdf'), bbox_inches="tight")



#---------
#You find the last snapshot before the spike simply by comparing the times of snapshots with the time of the spike sth like np.argmin((run.time-spiketime)**2)
spiketime = 0
#print(np.argmin((run.time-spiketime)**2))

#for i, t in enumerate(run.time):
#    print(f"{i}, {t/86400:.5}")
#-------------------------------------- 
fig, ax = plt.subplots(1,1)
ax.plot(run.time/86400, run.munbound/mass_gas)
ax.set_xlabel('Time / d')
ax.set_ylabel('Unbound mass, rel. envelope')    #(r'Unbound mass $M_\odot$')
fig.savefig('{}/{}'.format(plot_path, 'munbound.pdf'))

# you can add your own functions that the run can aggragate data from 
# first, define a function that takes an instance of the snapshot class as input, i.e. self
def get_magentic_energy(self):
    return ((self.bfld**2).sum(axis=1)/self.rho / (8.*np.pi)).sum()

# use addCallback to add the function to the run class, then aggregate
run.addCallback('emag', get_magentic_energy, 1) # name of quantity, function, number of dimensions
run.aggregateValues(['emag'])
print('Magnetic field amplification: {:.2e}'.format(run.emag[-1]/run.emag[0]))

# these quantities will be cached and recomputed so beware that if you change the functions or the snapshots are overwritten, the already computed values will remain until you delete them from the cache!
# to delete an item from the cache do something like this:
#item = 'emag'
#run.data['cached'].pop(item, None)
#run.data.pop(item, None)          
#run.saveCache()                   

# Good luck, have fun! If you have questions don't hesitate to ask anyone
