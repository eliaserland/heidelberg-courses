from typing import Coroutine
import numpy as np
from scipy import interpolate
from arepo_run import Run #/hits/fast/pso/olofsses/arepo-snap-util/
import matplotlib.pyplot as plt
plt.ioff()  # turn off plotting for other backends then saving
from const import *
import os

base_plotdir = '/hits/fast/pso/olofsses/5agb/ce-runs/'

models = {
	'5Magb_1.0corot_0.8rlof (copy)' : '/hits/fast/pso/olofsses/5agb/ce-runs/5Magb_copy/output',
	'5Magb_1.0corot_0.8rlof' : '/hits/fast/pso/olofsses/5agb/ce-runs/5Magb/output',
	'5Magb_0.0corot_0.8rlof' : '/hits/fast/pso/olofsses/5agb/ce-runs/5Magb_0.0corotation_0.8rlof/output',
	'5Magb_1.0corot_0.95rlof': '/hits/fast/pso/olofsses/5agb/ce-runs/5Magb_1.0corotation_0.95rlof/output',
}
#simulation_path = '/hits/fast/pso/olofsses/5agb/ce-runs/5Magb/output'
#simulation_path = '/hits/fast/pso/olofsses/5agb/ce-runs/5Magb_1.0corotation_0.95rlof/output'
#simulation_path = '/hits/fast/pso/olofsses/5agb/ce-runs/5Magb_0.0corotation_0.8rlof/output'
###simulation_path = '/hits/basement/pso/baumants/1.5M/ce-runs/1.5Mrgb_q0.50/output'

import matplotlib.font_manager
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"], # Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman
})


class Model:
	def __init__(self, model_name, simulation_path):
		if os.path.exists(simulation_path):
			self.model_name = model_name
			self.simulation_path = simulation_path
			self.plot_path = simulation_path + '/plots/other'
			if not os.path.exists(self.plot_path):
				print(f"Plot path '{self.plot_path}' does not exists, creating directory...")
				os.makedirs(self.plot_path)
			
			# use Run class from arepo_run to load the data			
			self.run = Run(self.simulation_path)
			# the Run class has a powerful cache system that lets you easily compute stuff accross all snapshots
			self.run.aggregateValues(['munbound', 'munboundthermal', 'munboundint', 'time'])
			
			# load the simulation data from here
			self.binary_file    = self.run.loadBinaryData()
			self.binary_ellipse = self.binary_file.get_ellipse_data()
			self.energy_file    = self.run.loadEnergyData()
			
			# Fix bug with inf pot energy of tracer particles, credit to Thomas.
			self.energy_file.epot = self.energy_file.epots[0] + self.energy_file.epots[1]               
			self.energy_file.etot = self.energy_file.epot + self.energy_file.ein + self.energy_file.ekin

			self.print_stuff()
		else:
			raise ValueError(f"ERROR: Simulation path '{simulation_path}' is not a valid directory.")        

	def plot_distance(self):
		""" Orbital separation distance (Rsol) and relative energy error over time (days). 
		"""
		fig, ax = plt.subplots(1,1)
		 
		ax.plot(self.binary_file.time/86400, self.binary_file.distance/rsol, label=tex_plain(self.model_name))
		ax.set_xlabel('Time / d')
		ax.set_ylabel(r'Distance / $R_\odot$')
		ax.legend(loc='upper center')
		plt.gca().set_ylim(bottom=0)
		 
		ax2 = ax.twinx()
		ax2.plot(self.energy_file.time/86400, np.abs((self.energy_file.etot - self.energy_file.etot[0])/self.energy_file.epot[0]), color='black')
		ax2.set_ylabel(r'$|(E_{tot}-E_{tot}[0])/E_{pot}[0]|$')
		plt.gca().set_ylim(bottom=0)
		
		fig.tight_layout()
		fig.savefig('{}/{}'.format(self.plot_path, self.model_name+'_distance.pdf'), bbox_inches="tight")
		plt.close('all')
	
	def plot_distance_energy_comps(self):
		""" Subplot 1: Orbital separation distance (Rsol) and relative energy error over time (days)
		    Subplot 2: Individual energy components (ergs) over time (days). 
		"""
		fig, ax = plt.subplots(2,1, figsize=(8,5)) #, sharex=True)
		ax = ax.flatten()

		# First subplot
		p1 = ax[0].semilogy(self.binary_file.time/86400, self.binary_file.distance/rsol, label='separation')
		ax0_twin = ax[0].twinx()
		p2 = ax0_twin.plot(self.energy_file.time/86400, 
						   (self.energy_file.etot - self.energy_file.etot[0])/self.energy_file.epot[0], 
						   label='rel. energy error', color='black')
		plt.gca().set_ylim(bottom=0)
		ax[0].set_xlabel('Time / d')
		ax[0].set_ylabel(r'Distance / $R_\odot$')
		ax0_twin.set_ylabel(r'$(E_{tot}-E_{tot}[0])/E_{pot}[0]$')

		lns = p1+p2
		labs = [l.get_label() for l in lns]
		ax[0].legend(lns, labs, loc=0)

		# Second subplot
		ax[1].plot(self.energy_file.time/86400, self.energy_file.etot, label='$E_{tot}$', color = 'black')
		ax[1].plot(self.energy_file.time/86400, self.energy_file.ein,  label='$E_{in}$')
		ax[1].plot(self.energy_file.time/86400, self.energy_file.epot, label='$E_{pot}$')
		ax[1].plot(self.energy_file.time/86400, self.energy_file.ekin, label='$E_{kin}$')

		ax[1].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
		ax[1].set_xlabel('Time / d')
		ax[1].set_ylabel(r'Energy / ergs')

		fig.tight_layout()
		fig.savefig('{}/{}'.format(self.plot_path, self.model_name+'_distance2.pdf'), bbox_inches="tight")
		plt.close('all')
	
	def plot_distance_snapshots(self):
		"""
		"""
		fig, ax = plt.subplots(1,1)
		 
		ax.semilogy(self.binary_file.time/86400, self.binary_file.distance/rsol)
		ax.set_xlabel('Time / d')
		ax.set_ylabel(r'Distance / $R_\odot$')
		 
		ax2 = ax.twinx()
		ax2.plot(self.energy_file.time/86400, self.energy_file.etot , color = 'black')
		#ax2.plot(energy_file.time/86400, energy_file.ekin) #, color = 'black')
		ax2.set_ylabel(r'Energy / ergs')

		#print(len(run.time), run.time.shape)
		#print(len(energy_file.etot), energy_file.etot.shape)
		#print(len(energy_file.time), energy_file.time.shape)
		#print(np.argmin((run.time-energy_file.time[:, None])**2, axis=0))

		# Time visualization for snapshots.
		ax2.plot(self.run.time/86400, self.energy_file.etot[np.argmin((self.run.time - self.energy_file.time[:, None])**2, axis=0)],
				 '.', label='snapshots', color='red')

		ax2.legend(loc='upper center')
		fig.savefig('{}/{}'.format(self.plot_path, self.model_name+'_distance3.pdf'), bbox_inches="tight")
		plt.close('all')

	def plot_unbound_mass(self):
		""" Plot unbound mass vs. time
		"""
		fig, ax = plt.subplots(1,1)
		ax.plot(self.run.time/86400, self.run.munbound/self.mass_gas)
		ax.set_xlabel('Time / d')
		ax.set_ylabel('Unbound mass, rel. envelope')    #(r'Unbound mass $M_\odot$')
		fig.savefig('{}/{}'.format(self.plot_path, self.model_name+'_munbound.pdf'))
		plt.close('all')
	
	def temperature_radial(self):
		""" Temperature vs. radial distance to primary for all gas particles.
		"""
		for snap_nr in range(self.run.snapcount):
			s = self.run.loadSnap(snap_nr) 
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
	
	def print_stuff(self):
		"""
		"""	
		# load a single snapshot like this:
		s = self.run.loadSnap(0) # loads snap 0.. 
		snap_cnt = self.run.snapcount
		#print(run.snapcount)

		# the snapshots contain data with arrays that have an entry for each particle in the simulation
		# access seperate particle types using the type array and masks
		print('Resolution: {:.2e} particles , containing {:.2e} gas particles, {:.2e} tracers, {} dm particles'.\
				format(len(s.type), len(s.type[s.type==0]), len(s.type[s.type==2]), len(s.type[s.type==1])))

		# s.data[]
		# s.type 0 1 2
		# Do not use angular momentum from the binary file, is it not correct.

		# check the mass distribution
		self.mass_gas = s.mass[s.type==0].sum() # = envolope mass
		mass_primary = s.mass[s.id==1e9][0] # the cores always have ids 1e9 and 1e9+1
		mass_companion = s.mass[s.id==1e9+1][0]
		print('Mass: {:.2f} in gas, {:.2f} in the primary core, {:.2f} in the companion core (in msun)'.\
			format(self.mass_gas/msol, mass_primary/msol, mass_companion/msol))
		
		# other quantities include vel, acce, time, soft, ...
		print('Simulation produced {} snapshots, simulation time at {:.2f}d'.format(self.run.snapcount, self.run.time[-1]/86400))
		

class Multi_Analysis:
	def __init__(self, models: dict, plotdir: str, groupname:str):
		self.plotdir = plotdir
		self.groupname = groupname
		self.models_dict = models	
		model_list = []
		for model_name, simulation_path in models.items():
			print(f"\nLoading model '{model_name}' at path '{simulation_path}'")
			model_list.append(Model(model_name, simulation_path))
		self.models_all = model_list
		self.models = self.models_all

	def set_model_selection(self, model_selection: list = None, groupname: str = None):
		if groupname is not None:
			self.groupname = groupname
		if model_selection is not None:
			models = []
			for i, key in enumerate(self.models_dict.keys()): # TODO: FIX THIS CRAP
				if any([key == model_str for model_str in model_selection]):
					models.append(self.models_all[i])
			self.models = models
		else:
			self.models = self.models_all
	
	def shift_time(self, timeshift: np.array):
		for i, m in enumerate(self.models_all):
			m.binary_file.time += timeshift[i]
			m.energy_file.time += timeshift[i]
			m.binary_ellipse.tsmooth += timeshift[i]
			m.binary_ellipse.tck_a = (m.binary_ellipse.tck_a[0] + timeshift[i],) + m.binary_ellipse.tck_a[1:]
			m.binary_ellipse.tck_b = (m.binary_ellipse.tck_b[0] + timeshift[i],) + m.binary_ellipse.tck_b[1:]

	def generate_all_individual_plots(self):
		for model in self.models:
			model.plot_distance()
			model.plot_distance_energy_comps()
			model.plot_distance_snapshots()
			model.plot_unbound_mass()
			#model.temperature_radial()
			
	def generate_cross_analysis_plots(self):
		self.plot_separation()
		self.plot_separation_log()
		self.plot_smooth_separation()
		self.plot_smooth_separation('semilogy')
		self.plot_smooth_separation('loglog')
	
	def plot_separation(self):
		""" Orbital separation distance (Rsol) + Total energy (ergs)  vs. Time (Days) 
		"""
		fig, ax = plt.subplots(1, 1, figsize=(8,6))

		for i, m in enumerate(self.models):	     
			ax.plot(m.binary_file.time/86400, m.binary_file.distance/rsol, label=tex_plain(m.model_name))    
		ax.set_xlabel('Time / d')
		ax.set_ylabel(r'Distance / $R_\odot$')
		ax.legend(loc='upper center')
		plt.gca().set_ylim(bottom=0)

		ax2 = ax.twinx()
		for i, m in enumerate(self.models):	     
			ax2.plot(m.energy_file.time/86400, 
						   np.abs((m.energy_file.etot - m.energy_file.etot[0])/m.energy_file.epot[0]), 
						   label='rel. energy error')
		ax2.set_ylabel(r'$|(E_{tot}-E_{tot}[0])/E_{pot}[0]|$')
		plt.gca().set_ylim(bottom=0)
		fig.tight_layout() 
		fig.savefig('{}/{}'.format(self.plotdir, self.groupname + '_distance_comparison.pdf'), bbox_inches="tight")
		plt.close('all')		


	def plot_separation_log(self):
		""" Orbital separation distance (Rsol) + Total energy (ergs)  vs. Time (Days) 
		"""
		fig, ax = plt.subplots(1, 1, figsize=(8,6))

		for i, m in enumerate(self.models):	     
			ax.semilogy(m.binary_file.time/86400, m.binary_file.distance/rsol, label=tex_plain(m.model_name))    
		ax.set_xlabel('Time / d')
		ax.set_ylabel(r'Distance / $R_\odot$')

		ax2 = ax.twinx()
		for i, m in enumerate(self.models):	     
			ax2.plot(m.energy_file.time/86400, 
						   np.abs((m.energy_file.etot - m.energy_file.etot[0])/m.energy_file.epot[0]), 
						   label=tex_plain(m.model_name))
		ax2.set_ylabel(r'$|(E_{tot}-E_{tot}[0])/E_{pot}[0]|$')
		ax2.legend(loc='upper center')
		fig.tight_layout()
		fig.savefig('{}/{}'.format(self.plotdir, self.groupname + '_distance_comparison_log.pdf'), bbox_inches="tight")
		plt.close('all')
	
	def plot_smooth_separation(self, scale: str = 'linear'):
		""" Orbital separation distance (Rsol) + Total energy (ergs)  vs. Time (Days)
		:param scale: 'linear', 'semilogy', 'loglog' 
		"""
		

		alpha_val_1 = 0.5
		alpha_val_2 = 1
		time_fraction = 0.55
		time_min = np.array([m.binary_file.time[-1] for m in self.models]).min()
		time_start = time_fraction * time_min

		fig, ax = plt.subplots(1, 1, figsize=(8,6))
		cmap = plt.get_cmap("tab10")

		for i, m in enumerate(self.models):
			ilo = np.argmin((time_start - m.binary_file.time)**2)
			ax.plot(m.binary_file.time[ilo:]/86400, m.binary_file.distance[ilo:]/rsol, 
				  color=cmap(i), linewidth=.6, alpha=alpha_val_1, label=tex_plain(m.model_name))

			ilo = np.argmin((time_start - m.binary_ellipse.tsmooth)**2)
			ax.plot(m.binary_ellipse.tsmooth[ilo:]/86400, 
						interpolate.splev(m.binary_ellipse.tsmooth, m.binary_ellipse.tck_a)[ilo:]/rsol, 
						color=cmap(i), label=tex_plain('a: B-spline'), alpha=alpha_val_2)
			ax.plot(m.binary_ellipse.tsmooth[ilo:]/86400, 
						interpolate.splev(m.binary_ellipse.tsmooth, m.binary_ellipse.tck_b)[ilo:]/rsol, 
						'-.' ,color=cmap(i), label=tex_plain('b: B-spline'), alpha=alpha_val_2)
			#ax.semilogy(m.binary_ellipse.tsmooth[ilo:]/86400, m.binary_ellipse.asmooth[ilo:]/rsol, label=tex_plain('asmooth '+ self.model_names[i]))
			#ax.semilogy(m.binary_ellipse.tsmooth[ilo:]/86400, m.binary_ellipse.bsmooth[ilo:]/rsol, label=tex_plain('bsmooth '+ self.model_names[i]))
			
		ax.set_xlabel('Time / d')
		ax.set_ylabel(r'Distance / $R_\odot$')
		ax.legend(loc='best') #'upper center'
		if scale == 'linear':
			plt.gca().set_ylim(bottom=0)
			fname = ''
		elif scale == 'semilogy':
			ax.set_yscale('log')
			fname = '_semilogy'
		elif scale == 'loglog':
			ax.set_xscale('log')
			ax.set_yscale('log')
			fname = 'loglog'
		else:
			raise ValueError(f"ERROR: Invalid keyword '{scale}'.")  
		fig.tight_layout()
		fig.savefig('{}/{}'.format(self.plotdir, self.groupname + '_distance_smooth_comp_' + scale + '.pdf'), bbox_inches="tight")
		plt.close('all')

	def plot_separation_semimajor(self):
		
		fig, ax = plt.subplots(2, 1, figsize=(8,6), sharex=True)
		ax = ax.flatten()
		
		for i, m in enumerate(self.models):	     
			ax[0].plot(m.binary_file.time/86400, m.binary_file.distance/rsol, label=tex_plain(m.model_name))    
		ax[0].set_ylabel(r'Distance / $R_\odot$')
		ax[0].legend(loc='upper center')
		plt.gca().set_ylim(bottom=0)
		"""
		ax2 = ax.twinx()
		for i, m in enumerate(self.models):	     
			ax2.plot(m.energy_file.time/86400, 
						   np.abs((m.energy_file.etot - m.energy_file.etot[0])/m.energy_file.epot[0]), 
						   label='rel. energy error')
		ax2.set_ylabel(r'$|(E_{tot}-E_{tot}[0])/E_{pot}[0]|$')
		"""
		time_fraction = 0.55
		time_min = np.array([m.binary_file.time[-1] for m in self.models]).min()
		time_start = time_fraction * time_min
		cmap = plt.get_cmap("tab10")
		for i, m in enumerate(self.models):
			#ilo = np.argmin((time_start - m.binary_file.time)**2)
			#ax.plot(m.binary_file.time[ilo:]/86400, m.binary_file.distance[ilo:]/rsol, 
			#	  color=cmap(i), linewidth=.6, alpha=alpha_val_1, label=tex_plain(m.model_name))

			ilo = np.argmin((time_start - m.binary_ellipse.tsmooth)**2)
			ax[1].plot(m.binary_ellipse.tsmooth[ilo:]/86400, 
						interpolate.splev(m.binary_ellipse.tsmooth, m.binary_ellipse.tck_a)[ilo:]/rsol, 
						color=cmap(i), label=tex_plain('a: B-spline'), alpha=alpha_val_2)
			ax[1].plot(m.binary_ellipse.tsmooth[ilo:]/86400, 
						interpolate.splev(m.binary_ellipse.tsmooth, m.binary_ellipse.tck_b)[ilo:]/rsol, 
						'-.' ,color=cmap(i), label=tex_plain('b: B-spline'), alpha=alpha_val_2)
			#ax.semilogy(m.binary_ellipse.tsmooth[ilo:]/86400, m.binary_ellipse.asmooth[ilo:]/rsol, label=tex_plain('asmooth '+ self.model_names[i]))
			#ax.semilogy(m.binary_ellipse.tsmooth[ilo:]/86400, m.binary_ellipse.bsmooth[ilo:]/rsol, label=tex_plain('bsmooth '+ self.model_names[i]))
			
		ax[1].set_xlabel('Time / d')	
		ax[1].set_ylabel(r'Distance / $R_\odot$')
		ax[1].legend(loc='best') #'upper center'

		plt.gca().set_ylim(bottom=0)
		fig.tight_layout() 
		fig.savefig('{}/{}'.format(self.plotdir, self.groupname + '_sep_semimajor.pdf'), bbox_inches="tight")
		plt.close('all')




	
def tex_plain(string):   
    if plt.rcParams['text.usetex']:
	    return '\\verb|' + string + '|' # in-line verbatim 
    else:
	    return string   
   

def main():
	# Load all models.
	a = Multi_Analysis(models, base_plotdir, '5Magb')
	
	# Apply timeshift
	manual_timeshift = np.array([0, 0, 0, -72*86400])
	a.shift_time(manual_timeshift)

	# All individual plots
	a.generate_all_individual_plots()

	# Full comparison between all models
	a.generate_cross_analysis_plots()

	# 0.8rlof comparison
	model_select = list(models.keys())
	model_select.remove('5Magb_1.0corot_0.95rlof')
	
	a.set_model_selection(model_select, '0.8rlof_comp')
	a.generate_cross_analysis_plots()

	# 5Magb 0.8rlof 1.0corot restart comparison
	model_select.remove('5Magb_0.0corot_0.8rlof')
	
	a.set_model_selection(model_select, '5Magb_restart')
	a.generate_cross_analysis_plots()
	

if __name__ == '__main__':
    main()

    #---------
#You find the last snapshot before the spike simply by comparing the times of snapshots with the time of the spike sth like np.argmin((run.time-spiketime)**2)
#spiketime = 0
#print(np.argmin((run.time-spiketime)**2))

#for i, t in enumerate(run.time):
#    print(f"{i}, {t/86400:.5}")
#--------------------------------------

"""
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
"""
    
