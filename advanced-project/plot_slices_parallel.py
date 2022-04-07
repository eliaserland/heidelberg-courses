#!/usr/bin/env python3

################################
################################
'''
This is a slight modification of plot_slices.py that
uses one core per snapshot to make it faster. The number of cores
can be given in models.py.

Melvin
'''

import sys, os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools

import multiprocessing as mp
from parallel_decorators import vectorize_parallel

from arepo_run import Run
from const import rsol, msol, NA, KB, G

from models import models, quantites, perspectives, seconds_per_day, POOLSIZE, NUMTHREADS, BOXSIZE, MAX_RHO, MIN_RHO, MIN_MACH, MAX_MACH, center_on_giant
from check_relaxation import eos, compute_velocity, compute_eos_stuff 
       
# output path for plots of slices - will be stored in this folder within the output folders of the Arepo runs
#NUMTHREADS = 1
output_path = 'plots/slices'

image_type = 'pdf' # Options: 'png', 'pdf'

# isotopes considered in the run
isotopes = ['h1', 'he4', 'c12', 'n14', 'o16']

BOXSIZE = BOXSIZE*rsol
RES = 2048
PLOT_KWARGS = dict(rasterized=True, shading='flat')

def plot_slices(models, output_path):
    for model in models:
        print('Plotting slices for model "%s"' % (model))

        path = os.path.join('./', model, 'output')
        if not os.path.exists(path):
            print('Cannot find model "'+model+'". Skipping...')
            continue

        save_path = os.path.join(path, output_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        run = Run(path)
        plot_snap(range(run.snapcount), run, save_path,)
      
      
@vectorize_parallel(method='processes', num_procs=POOLSIZE)
def plot_snap(idx, run, save_path):
	s = run.loadSnap(idx)

	calculated_mach_number = False

	for quantity, perspective in itertools.product(quantites, perspectives):
		snap_image_name = f'slice_{quantity}_{perspective}_{idx:03d}.'+image_type
		snap_image_path = os.path.join(save_path, snap_image_name)
		
		if os.path.exists(snap_image_path):
			print('   -   Skipping snapshot {:3.0f}, file \"{}\" already exists'.format(idx, snap_image_path))
			if quantity == 'mach_number':
				pass
				#calculated_mach_number = True
		else:
			print('   - Processing snapshot {:3.0f}, time = {:.2f} d, {} {}'.format(idx, s.time/seconds_per_day, quantity, perspective))	

			if quantity == 'mach_number' and not calculated_mach_number:
				# compute additional quantities for plotting
				compute_velocity(s)
				compute_eos_stuff(s, 1)
				s.data['mach_number'] = s.data['abs_vel'][s.type==0]/s.data['csnd']
				calculated_mach_number = True
			
			# plot stuff
			plot_and_save_slice(s, idx, quantity=quantity, save_path=save_path, perspective=perspective)			


def plot_and_save_slice(s, snap_num, quantity='rho', save_path='./', perspective='xy'):
   plt.rc('font', size=20)
   plt.rc('lines', linewidth=3.0)
   plt.rc('axes', linewidth=2.0)
   plt.rc('xtick', top=True)
   plt.rc('xtick', direction='out')
   plt.rc('xtick.major', pad=8)
   plt.rc('xtick.major', size=8.5)
   plt.rc('xtick.minor', size=5)
   plt.rc('xtick.major', width=2.0)
   plt.rc('xtick.minor', width=2.0)
   plt.rc('ytick', right=True)
   plt.rc('ytick', direction='out')
   plt.rc('ytick.major', pad=8)
   plt.rc('ytick.major', size=8.5)
   plt.rc('ytick.minor', size=5)
   plt.rc('ytick.major', width=2.0)
   plt.rc('ytick.minor', width=2.0)

   plt.rc('text', usetex=True)
   plt.rc('font', family='serif')


   fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12,12))
   cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)
   ax.set_aspect(1.0)

   axes, ax_labels = convert_axis(perspective)

   ax.set_xlabel(f'${{{ax_labels[0]}}}\,/\,\mathrm{{R}}_\odot$')
   ax.set_ylabel(f'${{{ax_labels[1]}}}\,/\,\mathrm{{R}}_\odot$')

   add_inner_title(ax, "{:.2f} d".format(s.time/seconds_per_day), color=get_color(quantity))

   image = plot_slice(s, ax, axes, quantity=quantity)

   cbar = fig.colorbar(image, cax=cax)
   cbar.ax.set_ylabel(get_cbar_label(quantity))

   fig.tight_layout()
   if image_type == 'png':
      fig.savefig(save_path + "/slice_%s_%s_%03d.png" % (quantity, perspective, snap_num), bbox_inches='tight', dpi=300)
   elif image_type == 'pdf':
      fig.savefig(save_path + "/slice_%s_%s_%03d.pdf" % (quantity, perspective, snap_num), bbox_inches='tight')
   plt.close(fig)


def convert_axis(perspective):
    xyz = set(['x', 'y', 'z'])
    str_len = len(perspective)
    # Verify correct input.
    if str_len == 2 and all([c in xyz for c in list(perspective)]):
        ax_labels = list(perspective)
        axes = []
        for c in perspective: 
            if c == 'x':
                axes.append(0)
            elif c == 'y':
                axes.append(1)
            elif c == 'z':
                axes.append(2)
        return axes, ax_labels
    else:
        raise ValueError("Incorrect argument 'perspective'. Specify two axis, e.g. 'xy' or 'zy'.")


def plot_slice(s, ax, axes, quantity='rho'):
   if center_on_giant and any(s.id == 1e9):
      center = s.pos[s.id == 1e9][0]
   else:
      center = s.centerofmass()
   rho_xy = s.get_Aslice(quantity,  box=[BOXSIZE, BOXSIZE],
                          center=center, axes=axes, res=RES,
                          grad=get_gradient(quantity), numthreads=NUMTHREADS)
   x_shifted = (rho_xy['x'] - center[axes[0]]) / rsol	
   y_shifted = (rho_xy['y'] - center[axes[1]]) / rsol
   image = ax.pcolormesh(x_shifted, y_shifted, rho_xy['grid'].T,
                          cmap=get_cmap(quantity), norm=get_norm(quantity),
                          **PLOT_KWARGS)
   if any(s.id == 1e9):
      giant_core = ( s.pos[s.id == 1e9][0] - center) / rsol
      ax.plot(giant_core[axes[0]], giant_core[axes[1]], marker='+', markersize=2*plt.rcParams['lines.markersize'], c='black')
   if any(s.id == 1e9 + 1):
      companion = ( s.pos[s.id == 1e9 + 1][0] - center) / rsol
      ax.plot(companion[axes[0]], companion[axes[1]], marker='x', markersize=2*plt.rcParams['lines.markersize'], c='white')
   
   ax.set_xlim(x_shifted.min(), x_shifted.max())
   ax.set_ylim(y_shifted.min(), y_shifted.max())
   ax.set_facecolor('k')
   return image


def add_inner_title(ax, title, loc=9, size=None, color='k', **kwargs):
   from matplotlib.offsetbox import AnchoredText
   if size is None:
      size = dict(size=plt.rcParams['legend.fontsize'], color=color)
   at = AnchoredText(title, loc=loc, prop=size, pad=0., borderpad=0.5, frameon=False, **kwargs)
   ax.add_artist(at)
   return at


def get_norm(quantity):
   norms = {'rho': mpl.colors.LogNorm(vmin=MIN_RHO, vmax=MAX_RHO),
            'mach_number': mpl.colors.Normalize(vmin=MIN_MACH, vmax=MAX_MACH),
            # 'mach_number': mpl.colors.LogNorm(vmin=1e-3, vmax=1e1),
            'pass00': mpl.colors.Normalize(vmin=0.0, vmax=1.0),
            'babs': mpl.colors.LogNorm(vmin=1e1, vmax=1e7),
            }
   return norms.get(quantity, None)


def get_color(quantity):
   colors = {'rho': 'w', 'mach_number': 'w'}
   return colors.get(quantity, 'w')


def get_cmap(quantity):
   cmap = {'rho': 'magma'}
   return cmap.get(quantity, 'magma')


def get_cbar_label(quantity):
   cbar_label = {'rho': r'$\rho\,/\,\mathrm{g}\,\mathrm{cm}^{-3}$',
                 'mach_number': r'Mach number $\mathcal{M}$',
                 }
   return cbar_label.get(quantity, None)

def get_gradient(quantity):
   gradients = {'rho': 'grar', 'pres': 'grap'}
   return gradients.get(quantity, None)


if __name__ == "__main__":
   plot_slices(models, output_path)
