from typing import Coroutine
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import copy
from scipy import interpolate
from arepo_run import Run
import matplotlib.pyplot as plt
plt.ioff()  # turn off plotting for other backends then saving
from const import *
import os, sys
from scipy.signal import savgol_filter

base_plotdir = '/hits/fast/pso/olofsses/5agb/ce-runs/plots'
file_ending  = 'pdf' # Options: 'pdf', 'eps', 'svg', 'png'
models = {
	'5Magb_1.0corot_0.8rlof' : ('/hits/fast/pso/olofsses/5agb/ce-runs/5Magb/output', 1.0, 0.8),
	'5Magb_0.0corot_0.8rlof' : ('/hits/fast/pso/olofsses/5agb/ce-runs/5Magb_0.0corotation_0.8rlof/output', 0.0, 0.8),
	'5Magb_1.0corot_0.95rlof': ('/hits/fast/pso/olofsses/5agb/ce-runs/5Magb_1.0corotation_0.95rlof/output', 1.0, 0.95),
}
#simulation_path = '/hits/fast/pso/olofsses/5agb/ce-runs/5Magb/output'
#simulation_path = '/hits/fast/pso/olofsses/5agb/ce-runs/5Magb_1.0corotation_0.95rlof/output'
#simulation_path = '/hits/fast/pso/olofsses/5agb/ce-runs/5Magb_0.0corotation_0.8rlof/output'
###simulation_path = '/hits/basement/pso/baumants/1.5M/ce-runs/1.5Mrgb_q0.50/output'

import matplotlib.font_manager
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["times", "times new roman"], # Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman
})


class Model:
	def __init__(self, model_name, simulation_path, corot, rlof):
		if os.path.exists(simulation_path):
			self.model_name = model_name
			self.simulation_path = simulation_path
			self.corot = corot
			self.rlof = rlof
			self.plot_path = simulation_path + '/plots/other'
			if not os.path.exists(self.plot_path):
				print(f"Plot path '{self.plot_path}' does not exists, creating directory...")
				os.makedirs(self.plot_path)
			self.timeshift = 0 

			# use Run class from arepo_run to load the data			
			self.run = Run(self.simulation_path)
			# the Run class has a powerful cache system that lets you easily compute stuff accross all snapshots
			self.run.aggregateValues(['munbound', 'munboundthermal', 'munboundint', 'time'])
			
			# load the simulation data from here
			self.binary_file    = self.run.loadBinaryData()
			self.calc_orbits()
			self.binary_ellipse = self.binary_file.get_ellipse_data()
			self.energy_file    = self.run.loadEnergyData()
			
			# Fix bug with inf pot energy of tracer particles, credit to Thomas.
			self.energy_file.epot = self.energy_file.epots[0] + self.energy_file.epots[1]               
			self.energy_file.etot = self.energy_file.epot + self.energy_file.ein + self.energy_file.ekin

			self.print_stuff()
		else:
			raise ValueError(f"ERROR: Simulation path '{simulation_path}' is not a valid directory.")        

	def calc_orbits(self):
		""" Calculate the no. of orbits for the binary system as a function of time.
		"""
		p = self.binary_file

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

		self.binary_file.n_orbits = dd_orbits['n']
		#return dd_orbits



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
		ax2.set_ylabel(r'$|(E_{\mathrm{tot}}-E_{\mathrm{tot}}[0])/E_{\mathrm{pot}}[0]|$')
		plt.gca().set_ylim(bottom=0)
		
		fig.tight_layout()
		fig.savefig('{}/{}'.format(self.plot_path, self.model_name+'_distance.'+file_ending), bbox_inches="tight")
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
						   np.abs((self.energy_file.etot - self.energy_file.etot[0])/self.energy_file.epot[0]), 
						   label='rel. energy error', color='black')
		plt.gca().set_ylim(bottom=0)
		ax[0].set_xlabel('Time / d')
		ax[0].set_ylabel(r'Distance / $R_\odot$')
		ax0_twin.set_ylabel(r'$|(E_{\mathrm{tot}}-E_{\mathrm{tot}}[0])/E_{\mathrm{pot}}[0]|$')

		lns = p1+p2
		labs = [l.get_label() for l in lns]
		ax[0].legend(lns, labs, loc='center left')

		# Second subplot
		ax[1].plot(self.energy_file.time/86400, self.energy_file.etot, label='$E_{\mathrm{tot}}$', color = 'black')
		ax[1].plot(self.energy_file.time/86400, self.energy_file.ein,  label='$E_{\mathrm{in}}$')
		ax[1].plot(self.energy_file.time/86400, self.energy_file.epot, label='$E_{\mathrm{pot}}$')
		ax[1].plot(self.energy_file.time/86400, self.energy_file.ekin, label='$E_{\mathrm{kin}}$')

		ax[1].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
		ax[1].set_xlabel('Time / d')
		ax[1].set_ylabel(r'Energy / ergs')

		fig.tight_layout()
		fig.savefig('{}/{}'.format(self.plot_path, self.model_name+'_distance2.'+file_ending), bbox_inches="tight")
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
		ax2.set_ylabel(r'Energy / ergs')
		
		# Time visualization for snapshots.
		ax2.plot(self.run.time/86400, self.energy_file.etot[np.argmin((self.run.time - self.energy_file.time[:, None])**2, axis=0)],
				 '.', label='snapshots', color='red')

		ax2.legend(loc='upper center')
		fig.tight_layout() 
		fig.savefig('{}/{}'.format(self.plot_path, self.model_name+'_distance3.'+file_ending), bbox_inches="tight")
		plt.close('all')

	def plot_unbound_mass(self):
		""" Plot unbound mass vs. time
		"""
		s = self.run.loadSnap(0) # load snap 0
		self.mass_gas = s.mass[s.type==0].sum() # total mass of envolope
		
		fig, ax = plt.subplots(1,1)

		# Filter out snapshots past the current timepoint of the simulation.
		mask = self.run.time < self.binary_file.time[-1]
		
		# Orbital separation:
		ax.plot(self.binary_file.time/86400, self.binary_file.distance/rsol, 
			color='darkgray', zorder=0)
		ax.set_xlabel('Time / d')
		ax.set_ylabel('Distance / $R_\odot$') #(r'Unbound mass $M_\odot$') #, rel. envelope'
		ax.set_xlim(left=0)
		ax.set_ylim(bottom=0)

		# Unbound mass, 3 criteria: 'munbound', 'munboundthermal', 'munboundint'
		normal_label   = '$e_{\mathrm{kin}} + e_{\mathrm{pot}} > 0$'
		thermal_label  = '$e_{\mathrm{kin}} + e_{\mathrm{pot}} + e_{\mathrm{th}}> 0$'
		internal_label = '$e_{\mathrm{kin}} + e_{\mathrm{pot}} + e_{\mathrm{int}}> 0$'
		ax2 = ax.twinx()
		ax2.plot(self.run.time[mask]/86400, self.run.munbound[mask]/self.mass_gas, label=normal_label, zorder=1) # ax.plot(self.run.time/86400, self.run.munbound/self.mass_gas)
		ax2.plot(self.run.time[mask]/86400, self.run.munboundthermal[mask]/self.mass_gas, label=thermal_label, zorder=2)
		ax2.plot(self.run.time[mask]/86400, self.run.munboundint[mask]/self.mass_gas, label=internal_label, zorder=3)
		ax2.legend(loc='center left', frameon=False)
		ax2.set_ylim(bottom=0)
		ax2.set_ylabel('$M_{\mathrm{ej}}$ / $M_{\mathrm{tot}}$')

		# Number of completed orbits (top x-axis):
		n_orbits = self.binary_file.n_orbits
		time 	 = self.binary_file.time
		orbits_arr = np.array([0, 1, 2, 3, 5, 10, 25, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000])
		orbits_arr = orbits_arr[orbits_arr <= np.max(n_orbits)]              # Filter out markers we won't need.
		orbits_idx = np.argmin(abs(n_orbits - orbits_arr[:, None]), axis=1) # Get indices closest to each marker.
		x_ticks = list(time[orbits_idx]/86400)
		x_labels = [str(i) for i in list(orbits_arr.astype(int))]
		axtop = ax.twiny()
		axtop.set_xticks(x_ticks)
		axtop.set_xticklabels(x_labels)
		axtop.set_xlabel('Number of orbits / $\mathrm{N}$')
		axtop.set_xlim(ax.get_xlim())

		fig.tight_layout() 
		fig.savefig('{}/{}'.format(self.plot_path, self.model_name+'_munbound.'+file_ending))
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
			fig.tight_layout() 
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
		for model_name, (simulation_path, corot, rlof) in models.items():
			print(f"\nLoading model '{model_name}' at path '{simulation_path}'")
			model_list.append(Model(model_name, simulation_path, corot, rlof))
		self.models_all = model_list
		self.models = self.models_all
		self.saved_selection = None
		self.timeshift_is_set = False
		self.timeshift_is_applied = False

	def set_model_selection(self, model_selection: list = None, groupname: str = None, reset_selection=False):
		if groupname is not None:
			self.groupname = groupname
		if reset_selection is True:
			self.models = self.models_all
		else:
			# Create selection
			if model_selection is not None:
				self.saved_selection = model_selection
			elif self.saved_selection is None:				
				raise ValueError(f"ERROR: No provided or saved model selection.") 
			models = []
			for i, key in enumerate(self.models_dict.keys()):
				if any([key == model_str for model_str in self.saved_selection]):						
					models.append(self.models_all[i])
			self.models = models
	
	def set_timeshift(self, timeshift: np.array):
		if self.timeshift_is_set: 
			if self.timeshift_is_applied:
				self.reset_timeshift()
		else:
			self.timeshift_is_set = True
		self.timeshift = timeshift
		self.apply_timeshift()

	def apply_timeshift(self):
		if self.timeshift_is_set and not self.timeshift_is_applied:
			self.shift_time(self.timeshift)
			self.timeshift_is_applied = True
			self.set_model_selection()
			for i, m in enumerate(self.models_all):
				m.timeshift = self.timeshift[i]

	def reset_timeshift(self):
		if self.timeshift_is_set and self.timeshift_is_applied:
			self.shift_time(-self.timeshift)
			self.timeshift_is_applied = False
			self.set_model_selection()
			for i, m in enumerate(self.models_all):
				m.timeshift = 0

	def shift_time(self, timeshift: np.array):
		for i, m in enumerate(self.models_all):
			m.binary_file.time += timeshift[i]
			m.energy_file.time += timeshift[i]
			m.binary_ellipse.tsmooth += timeshift[i]
			m.binary_ellipse.tck_a = (m.binary_ellipse.tck_a[0] + timeshift[i],) + m.binary_ellipse.tck_a[1:]
			m.binary_ellipse.tck_b = (m.binary_ellipse.tck_b[0] + timeshift[i],) + m.binary_ellipse.tck_b[1:]
	

	def generate_all_individual_plots(self):
		print(f'\nGenerating individual plots...')
		for model in self.models:
			print(f'Model: {model.model_name}')
			model.plot_distance()
			model.plot_distance_energy_comps()
			model.plot_distance_snapshots()
			model.plot_unbound_mass()
			#model.temperature_radial()
		print('Done!')
			
	def generate_cross_analysis_plots(self):
		print(f'\nGroup: {self.groupname}, generating cross analysis plots... ')
		
		# Bad plots
		self.plot_multi_separation()
		self.plot_multi_separation('log')
		self.plot_multi_smooth_separation()
		self.plot_multi_smooth_separation('semilogy')
		self.plot_multi_smooth_separation('loglog')
		self.plot_multi_separation_semimajor_TEST(show_energy_err=True)
		
		# Nice plots  
		self.plot_relative_energy_err()
		self.plot_multi_separation_semimajor(use_timeshift=False)
		self.plot_multi_separation_semimajor(use_timeshift=True)
		self.plot_multi_unbound_mass()
		
		print('Done!')
	
	def plot_multi_separation(self, scale: str = 'linear'):
		""" Orbital separation distance (Rsol) + Total energy (ergs)  vs. Time (Days) 
		:param scale: 'linear', 'log' 
		"""
		fig, ax = plt.subplots(1, 1, figsize=(8,6))

		for i, m in enumerate(self.models):	     
			ax.plot(m.binary_file.time/86400, m.binary_file.distance/rsol, label=tex_plain(m.model_name))    
		ax.set_xlabel('Time / d')
		ax.set_ylabel(r'Distance / $R_\odot$')
		ax.legend(loc='upper center', frameon=False)
		if scale == 'linear':
			plt.gca().set_ylim(bottom=0)
		elif scale == 'log':
			ax.set_yscale('log')
		else:
			raise ValueError(f"ERROR: Invalid keyword '{scale}'.")

		ax2 = ax.twinx()
		for i, m in enumerate(self.models):	     
			ax2.plot(m.energy_file.time/86400, np.abs((m.energy_file.etot - m.energy_file.etot[0])/m.energy_file.epot[0]), 
				 label='rel. energy error')
		ax2.set_ylabel(r'$|(E_{\mathrm{tot}}-E_{\mathrm{tot}}[0])/E_{\mathrm{pot}}[0]|$')
		plt.gca().set_ylim(bottom=0)
		
		fig.tight_layout() 
		fig.savefig('{}/{}'.format(self.plotdir, self.groupname + '_distance_comparison_'+scale+'.'+file_ending), bbox_inches="tight")
		plt.close('all')		

	
	def plot_multi_smooth_separation(self, scale: str = 'linear'):
		""" Orbital separation distance (Rsol) + Total energy (ergs)  vs. Time (Days)
		:param scale: 'linear', 'semilogy', 'loglog' 
		"""
		alpha_val_1 = 0.5
		alpha_val_2 = 1
		time_fraction = 0.6
		time_spans = np.array([m.binary_file.time[-1]-m.binary_file.time[0] for m in self.models])
		time_start = time_fraction * np.min(time_spans) + self.models[np.argmin(time_spans)].binary_file.time[0]

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
		ax.legend(loc='best', frameon=False) #'upper center'
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
		fig.savefig('{}/{}'.format(self.plotdir, self.groupname+'_distance_smooth_comp_'+scale+'.'+file_ending), bbox_inches="tight")
		plt.close('all')

	def plot_multi_separation_semimajor(self, use_timeshift=False):
		""" Plot orbital separation and smoothened semi-major axis over time, 
		    with or without timeshift.
		"""
		fig, ax = plt.subplots(2, 1, figsize=(8,6), sharex=True) #figsize=(8,6)
		ax = ax.flatten()
		
		# Binary separation
		for i, m in enumerate(self.models):
			if not use_timeshift: self.reset_timeshift()
			if abs(m.timeshift) > 0:
				timeshift_str = f' (offset +{int(m.timeshift/86400)} d)'
			else:
				timeshift_str = ''
			ax[0].plot(m.binary_file.time/86400, m.binary_file.distance/rsol, 
				   label=f'$\chi = {m.corot}$,\ \ $a_i = {m.rlof}\ R_{{\mathrm{{RLOF}}}}$'+timeshift_str)
			if not use_timeshift: self.apply_timeshift()    
		ax[0].set_ylabel(r'Distance / $R_\odot$')
		if use_timeshift: loc = 'center left'
		else: loc = 'best'
		ax[0].legend(loc=loc, frameon=False)
		ax[0].set_ylim(bottom=0)
		ax[0].set_xlim(left=0)

		# Smoothened semi-major axis
		time_fraction = 0.6
		time_spans = np.array([m.binary_file.time[-1]-m.binary_file.time[0] for m in self.models])
		time_start = time_fraction * np.min(time_spans) + self.models[np.argmin(time_spans)].binary_file.time[0]
		cmap = plt.get_cmap("tab10")
		for i, m in enumerate(self.models):
			ilo = np.argmin((time_start - m.binary_ellipse.tsmooth)**2)
			if not use_timeshift: self.reset_timeshift()
			if abs(m.timeshift) > 0:
				timeshift_str = f' (offset +{int(m.timeshift/86400)} d)'
			else:
				timeshift_str = ''
			ax[1].plot(m.binary_ellipse.tsmooth[ilo:]/86400, interpolate.splev(m.binary_ellipse.tsmooth, m.binary_ellipse.tck_a)[ilo:]/rsol, 
				   color=cmap(i), label=f'$\chi = {m.corot}$,\ \ $a_i = {m.rlof}\ R_{{\mathrm{{RLOF}}}}$'+timeshift_str)
			if use_timeshift:
				print(f'Model {m.model_name:23} final separation: {np.min(interpolate.splev(m.binary_ellipse.tsmooth, m.binary_ellipse.tck_a)[ilo:]/rsol):.5} Rsol')
				print(f'Model {m.model_name:23} final/init ratio: {np.min(interpolate.splev(m.binary_ellipse.tsmooth, m.binary_ellipse.tck_a)[ilo:])/m.binary_file.distance[0]:.5}')

			if not use_timeshift: self.apply_timeshift()	
		ax[1].set_ylabel(r'Semi-major axis / $R_\odot$')
		ax[1].legend(loc='center left', frameon=False) #'upper center'
		ax[1].set_xlim(ax[0].get_xlim())
		ax[1].set_ylim(bottom=0) #set_ylim(ax[0].get_ylim()) #set_ylim(bottom=0)
		ax[1].set_xlabel('Time / d')

		if use_timeshift: 
			filename_ending = '_semimajor_timeshifted.'+file_ending
		else: 
			filename_ending = '_semimajor.'+file_ending
		fig.tight_layout() 
		fig.savefig('{}/{}'.format(self.plotdir, self.groupname + filename_ending), bbox_inches="tight")

	def plot_relative_energy_err(self, use_timeshift=False):
		fig, ax = plt.subplots(1, 1, figsize=(8,4)) 
		#ax = ax.flatten()
		if not use_timeshift: self.reset_timeshift()
		for i, m in enumerate(self.models):
			if abs(m.timeshift) > 0:
				timeshift_str = f' (offset +{int(m.timeshift/86400)} d)'
			else:
				timeshift_str = ''

			rel_energy_err = np.abs((m.energy_file.etot - m.energy_file.etot[0])/m.energy_file.epot[0])
			ax.plot(m.energy_file.time/86400, rel_energy_err, 
				label=f'$\chi = {m.corot}$,\ \ $a_i = {m.rlof}\ R_{{\mathrm{{RLOF}}}}$'+timeshift_str)
		
			print(f'Model {m.model_name} relative energy error: {np.max(rel_energy_err)*100:.2}% ({np.max(rel_energy_err)*100:.5}%)')
		
		if not use_timeshift: self.apply_timeshift()
		ax.set_ylabel(r'$|(E_{\mathrm{tot}}-E_{\mathrm{tot}, 0})/E_{\mathrm{pot}, 0}|$') # r'$|(E_{tot}-E_{tot}[0])/E_{pot}[0]|$'
		ax.set_ylim(bottom=0)
		ax.set_xlim(left=0)
		ax.legend(loc='best', frameon=False)
		ax.set_xlabel('Time / d')
		fig.tight_layout() 
		fig.savefig('{}/{}'.format(self.plotdir, self.groupname + '_energy_err.'+file_ending), bbox_inches="tight")
	
	def plot_multi_separation_semimajor_TEST(self, show_energy_err=False):
		if show_energy_err:
			no_subplots = 4
		else:
			no_subplots = 3
		fig, ax = plt.subplots(no_subplots, 1, figsize=(8,2*6), sharex=True)
		ax = ax.flatten()
		self.reset_timeshift()
		for i, m in enumerate(self.models):	     
			ax[0].plot(m.binary_file.time/86400, m.binary_file.distance/rsol, label=tex_plain(m.model_name))    
		self.apply_timeshift()
		ax[0].set_ylabel(r'Distance / $R_\odot$')
		ax[0].legend(loc='center left', frameon=False)
		ax[0].set_ylim(bottom=0)

		alpha_val_2 = 1
		time_fraction = 0.6
		time_spans = np.array([m.binary_file.time[-1]-m.binary_file.time[0] for m in self.models])
		time_start = time_fraction * np.min(time_spans) + self.models[np.argmin(time_spans)].binary_file.time[0]
		cmap = plt.get_cmap("tab10")
		for i, m in enumerate(self.models):
			ilo = np.argmin((time_start - m.binary_ellipse.tsmooth)**2)
			self.reset_timeshift()
			ax[1].plot(m.binary_ellipse.tsmooth[ilo:]/86400, interpolate.splev(m.binary_ellipse.tsmooth, m.binary_ellipse.tck_a)[ilo:]/rsol, 
				   color=cmap(i), label=tex_plain(m.model_name), alpha=alpha_val_2)
			#print(f'Final/initial separation {m.model_name}: {interpolate.splev(m.binary_ellipse.tsmooth, m.binary_ellipse.tck_a)[-1]/rsol:.5} Rsol, {m.binary_file.distance[0]/rsol:.5} Rsol.')	
			self.apply_timeshift()
		ax[1].set_ylabel(r'Semi-major axis / $R_\odot$')
		ax[1].legend(loc='center left', frameon=False) #'upper center'
		ax[1].set_xlim(ax[0].get_xlim())
		ax[1].set_ylim(bottom=0) #set_ylim(ax[0].get_ylim()) #set_ylim(bottom=0)


		for i, m in enumerate(self.models):
			ilo = np.argmin((time_start - m.binary_ellipse.tsmooth)**2)

			t = m.binary_ellipse.tsmooth
			tck_a = m.binary_ellipse.tck_a
			a_dot_over_a = (interpolate.splev(t, tck_a, der=1)/interpolate.splev(t, tck_a))
			adota_filtered = savgol_filter(a_dot_over_a, 31, 2)
			adota_filtered = savgol_filter(a_dot_over_a, 101, 2)
			adota_filtered = savgol_filter(adota_filtered, 351, 2)
			adota_filtered = savgol_filter(adota_filtered, 801, 2)
			offset = 500

			#ax[2].plot(t[ilo:]/86400, interpolate.splev(t, tck_a, der=1)[ilo:]/rsol, 
			#	   color=cmap(i), label=tex_plain(m.model_name), alpha=alpha_val_2)
			ax[2].plot(t[ilo+offset:]/86400, adota_filtered[ilo+offset:], 
				   color=cmap(i), label=tex_plain(m.model_name), alpha=alpha_val_2)
			ax[2].plot(t[ilo+offset:]/86400, a_dot_over_a[ilo+offset:], 
				   color=cmap(i), label=tex_plain(m.model_name), alpha=0.5, linewidth=0.5)
		ax[2].set_ylabel(r'$\dot{a}/a$')
		ax[2].set_ylim(top=0, bottom=-1e-6)
		ax[2].legend(loc='center left', frameon=False)

		if show_energy_err:
			for i, m in enumerate(self.models):
				ax[-1].plot(m.energy_file.time/86400, np.abs((m.energy_file.etot - m.energy_file.etot[0])/m.energy_file.epot[0]), 
					label=tex_plain(m.model_name))
			ax[-1].set_ylabel(r'$|(E_{\mathrm{tot}}-E_{\mathrm{tot}}[0])/E_{\mathrm{pot}}[0]|$') # r'$|(E_{tot}-E_{tot}[0])/E_{pot}[0]|$'
			ax[-1].set_ylim(bottom=0)
			ax[-1].legend(loc='center left', frameon=False)
			ax[-1].set_xlabel('Time / d')
			filename_ending = '_sep_semimajor_energy_adota.'+file_ending
		else:
			ax[-1].set_xlabel('Time / d')
			filename_ending = '_sep_semimajor.'+file_ending

		fig.tight_layout() 
		fig.savefig('{}/{}'.format(self.plotdir, self.groupname + filename_ending), bbox_inches="tight")
		plt.close('all')

	def plot_multi_unbound_mass(self):
		""" Plot unbound mass vs. time
		"""
		self.reset_timeshift() # If applied, remove timeshift

		normal_label   = '$e_{\mathrm{kin}} + e_{\mathrm{pot}} > 0$'
		thermal_label  = '$e_{\mathrm{kin}} + e_{\mathrm{pot}} + e_{\mathrm{th}}> 0$'
		internal_label = '$e_{\mathrm{kin}} + e_{\mathrm{pot}} + e_{\mathrm{int}}> 0$'
	
		fig, ax = plt.subplots(len(self.models), 1, figsize=(8, 8))
		ax  = ax.flatten()
		ax2 = np.array([ax[i].twinx() for i in range(len(self.models))])

		for i, m in enumerate(self.models):
			s = m.run.loadSnap(0) 		     # load snap 0
			m.mass_gas = s.mass[s.type==0].sum() # calculate envolope mass		
			
			# Orbital separation:
			ax[i].plot(m.binary_file.time/86400, m.binary_file.distance/rsol, 
				   color='darkgray', zorder=0)
			#ax[i].set_title(tex_plain(m.model_name))
			ax[i].set_xlabel('Time / d')	
			ax[i].set_ylabel('Distance / $R_\odot$') 
			ax[i].set_xlim(left=0)
			ax[i].set_ylim(bottom=0)
			
			# Filter out snapshots past the current timepoint of the simulation.
			mask = m.run.time <= m.binary_file.time[-1]
			
			# Unbound mass, 3 criteria: 'munbound', 'munboundthermal', 'munboundint'
			ax2[i].plot(m.run.time[mask]/86400, m.run.munbound[mask]/m.mass_gas, label=normal_label, zorder=1)
			ax2[i].plot(m.run.time[mask]/86400, m.run.munboundthermal[mask]/m.mass_gas, label=thermal_label, zorder=2)
			ax2[i].plot(m.run.time[mask]/86400, m.run.munboundint[mask]/m.mass_gas, label=internal_label, zorder=3)
			ax2[i].legend(loc='center left', frameon=False, title=f'$\chi = {m.corot}$,\ \ \ $a_i = {m.rlof}\ R_{{\mathrm{{RLOF}}}}$') #5M_\odot$ AGB,\\ $q = 0.25$\n
			ax2[i].set_ylabel('$M_{\mathrm{ej}}$ / $M_{\mathrm{tot}}$')
			ax2[i].set_ylim(bottom=0)

			print(f'Model {m.model_name:23}, rel. unbound mass (kinetic criterion): {np.max(m.run.munbound[mask]/m.mass_gas)*100:.3} %')

			# Number of completed orbits (top x-axis):
			n_orbits = m.binary_file.n_orbits
			time 	 = m.binary_file.time
			orbits_arr = np.array([0, 1, 2, 3, 5, 10, 25, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000])
			orbits_arr = orbits_arr[orbits_arr <= np.max(n_orbits)]              # Filter out markers we won't need.
			orbits_idx = np.argmin(abs(n_orbits - orbits_arr[:, None]), axis=1) # Get indices closest to each marker.
			x_ticks = list(time[orbits_idx]/86400)
			x_labels = [str(i) for i in list(orbits_arr.astype(int))]
			axtop = ax[i].twiny()
			axtop.set_xticks(x_ticks)
			axtop.set_xticklabels(x_labels)
			axtop.set_xlabel('Number of orbits / $\mathrm{N}$')
			axtop.set_xlim(ax[i].get_xlim())
		

		fig.tight_layout() 
		fig.savefig('{}/{}'.format(self.plotdir, self.groupname +'_munbound.'+file_ending), bbox_inches="tight")
		self.apply_timeshift() # If set, apply timeshift

	


def tex_plain(string):   
    if plt.rcParams['text.usetex']:
	    return '\\verb|' + string + '|' # in-line verbatim 
    else:
	    return string   
   
def main():
	# Load all models.
	a = Multi_Analysis(models, base_plotdir, '5Magb')
	
	# All individual plots
	a.generate_all_individual_plots()
	
	# Full comparison between all models
	#model_select = list(models.keys())
	#model_select.remove('5Magb_1.0corot_0.8rlof (copy)')
	#a.set_model_selection(model_select, '5Magb')
	manual_timeshift = np.array([72*86400, 72*86400, 72*86400, 0]) #manual_timeshift = np.array([0, 0, 0, -72*86400])
	a.set_timeshift(manual_timeshift) # Apply timeshift		
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
    
