#!/usr/bin/env python

import sys, os
import numpy as np
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

import mesa as ms
from arepo_run import Run
from const import rsol, msol, NA, KB, G
from stellar_ics.tools import disable_stdout
from stellar_ics.eoswrapper import OpalEOS
eos = OpalEOS()

from models import models, seconds_per_day, NUMTHREADS, MIN_RADIUS, MAX_RADIUS

# number of time-snapshots to plot in each panel
number_times = 4

# output path for plots
output_path = './plots'

# plotting range
range = np.array([MIN_RADIUS, MAX_RADIUS], dtype=np.float64)*rsol

# isotopes considered in the run
isotopes = ['h1', 'he4', 'c12', 'n14', 'o16']

# how many processes shall we use in parallel?
number_processes = NUMTHREADS

def check_relaxation(models, plot_path='./plots', number_processes=2):
   for model in models:
      print('Checking model "%s"' % (model))
      # init Arepo Run of model
      path = os.path.join('./', model, 'output')
      if not os.path.exists(path):
         print('Cannot find model "'+model+'". Skipping...')
         continue

      # does the plot path exists? If not, create it
      if not os.path.exists(plot_path):
         os.mkdir(plot_path)

      quantities = {}
      run = Run(path, nproc=number_processes)
      indeces = np.linspace(start=0, stop=run.snapcount-1, num=number_times, dtype=int)
      for idx in indeces:
         s = run.loadSnap(idx)

         print('   - Processing snapshot {:.0f}, time = {:.2f} d'.format(idx, s.time/seconds_per_day))

         # First of all, get rid of dark matter particle
         s.mass = s.mass[s.type==0]
         s.pos = s.pos[s.type==0]
         s.vel = s.vel[s.type==0]
         s.data['acce'] = s.data['acce'][s.type==0]


         # second, compute and set all required quantities
         compute_radius(s)
         set_isotope_data(s)
         compute_velocity(s)
         compute_eos_stuff(s, number_processes)
         s.data['mach_number'] = s.data['abs_vel']/s.data['csnd']
         #s.data['entr'] = np.ones_like(s.data['rho'])
         #s.data['mach_number'] = np.random.rand(len(s.data['rho']))
         compute_accelerations(s)

         qnames = ['rho', 'temp', 'entr', 'mach_number', 'abs_grav_acc', 'abs_grad_p', 'rel_dacc_norm']
         quantities[idx] = {'time': s.time, 'qs': compute_1d_quantities(s, qnames=qnames, range=range, nbins=None)}

      # finally, make plots of the computed quantities
      plot_1d_quantities(quantities, mesa_profile=models[model]['mesa_profile'], basename=os.path.join(plot_path, model), range=range)


def plot_1d_quantities(quantities, mesa_profile, basename='model-name', range=None):
   import matplotlib.pyplot as plt
   from cycler import cycler

   # my cyclers for these plots
   #                                  blue     orange    greenish    reddish   sky blue     black     yellow     purple
   default_cycler = (cycler(color=['#0072B2', '#E69F00', '#009E73', '#D55E00', '#56B4E9', '#000000', '#F0E442', '#CC79A7']) +
                     cycler(linestyle=['-', '--', ':', '-.', '-', '--', ':', '-.']))
   hydroeq_cycler = (cycler(color=['#0072B2', '#E69F00', '#009E73', '#D55E00', '#56B4E9', '#000000', '#F0E442', '#CC79A7']) *
                     cycler(linestyle=['-', '--']))

   # some standard plot changes
   #plt.rc('axes', prop_cycle=default_cycler)
   plt.rc('font', size=20)
   plt.rc('lines', linewidth=3.0)
   plt.rc('axes', linewidth=2.0)
   plt.rc('xtick', top=True)
   plt.rc('xtick', direction='in')
   plt.rc('xtick.major', pad=8)
   plt.rc('xtick.major', size=8.5)
   plt.rc('xtick.minor', size=5)
   plt.rc('xtick.major', width=2.0)
   plt.rc('xtick.minor', width=2.0)
   plt.rc('ytick', right=True)
   plt.rc('ytick', direction='in')
   plt.rc('ytick.major', pad=8)
   plt.rc('ytick.major', size=8.5)
   plt.rc('ytick.minor', size=5)
   plt.rc('ytick.major', width=2.0)
   plt.rc('ytick.minor', width=2.0)

   fig1, ax1 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12,14)) # Mach number and density
   fig2, ax2 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12,14)) # temperature and entropy
   fig3, ax3 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12,14)) # hydro equilibrium

   # equivalent original MESA data
   get_mesa_profile = disable_stdout(ms.mesa_profile)
   pro = get_mesa_profile('./', num_type='explicit', give_filename=mesa_profile)
   rmid = pro.get('rmid')
   rho = np.power(10, pro.get('logRho'))
   mach = pro.get('conv_vel_div_csound')
   temp = np.power(10, pro.get('logT'))
   entr = pro.get('entropy')*NA*KB/1e9
   #ax1[0].loglog(rmid, mach, label='MESA', ls='-', lw=11, c='0.65')
   ax1[1].loglog(rmid, rho, label='MESA', ls='-', lw=11, c='0.65')
   ax2[0].semilogx(rmid, entr, label='MESA', ls='-', lw=11, c='0.65')
   ax2[1].loglog(rmid, temp, label='MESA', ls='-', lw=11, c='0.65')

   ax1[0].set_prop_cycle(default_cycler)
   ax1[1].set_prop_cycle(default_cycler)
   ax2[0].set_prop_cycle(default_cycler)
   ax2[1].set_prop_cycle(default_cycler)
   ax3[0].set_prop_cycle(hydroeq_cycler)
   ax3[1].set_prop_cycle(default_cycler)

   # relaxed Arepo models
   for idx in quantities:
      # averaged Arepo data
      ax1[0].loglog( quantities[idx]['qs']['mach_number']['radius']/rsol, 
                     quantities[idx]['qs']['mach_number']['values'], 
                     label='t = %.2f d' % (quantities[idx]['time']/seconds_per_day))
      ax1[1].loglog( quantities[idx]['qs']['rho']['radius']/rsol, 
                     quantities[idx]['qs']['rho']['values'], 
                     label='t = %.2f d' % (quantities[idx]['time']/seconds_per_day))
      ax2[0].semilogx(  quantities[idx]['qs']['entr']['radius']/rsol, 
                        quantities[idx]['qs']['entr']['values']/1e9, 
                        label='t = %.2f d' % (quantities[idx]['time']/seconds_per_day))
      ax2[1].loglog( quantities[idx]['qs']['temp']['radius']/rsol, 
                     quantities[idx]['qs']['temp']['values'], 
                     label='t = %.2f d' % (quantities[idx]['time']/seconds_per_day))
      ax3[0].loglog( quantities[idx]['qs']['abs_grav_acc']['radius']/rsol, 
                     quantities[idx]['qs']['abs_grav_acc']['values'], 
                     label=r'$|\rho\mathbf{g}|$, t = %.2f d' % (quantities[idx]['time']/seconds_per_day))
      ax3[0].loglog( quantities[idx]['qs']['abs_grad_p']['radius']/rsol, 
                     quantities[idx]['qs']['abs_grad_p']['values'], 
                     label=r'$|\nabla P|$, t = %.2f d' % (quantities[idx]['time']/seconds_per_day))
      ax3[1].loglog( quantities[idx]['qs']['rel_dacc_norm']['radius']/rsol, 
                     quantities[idx]['qs']['rel_dacc_norm']['values'], 
                     label='t = %.2f d' % (quantities[idx]['time']/seconds_per_day))

   ax1[1].set_xlabel(r'$r\,/\,\mathrm{R}_\odot$')
   ax2[1].set_xlabel(r'$r\,/\,\mathrm{R}_\odot$')
   ax3[1].set_xlabel(r'$r\,/\,\mathrm{R}_\odot$')

   ax1[0].set_ylabel(r'Mach number $\mathcal{M}$')
   ax1[1].set_ylabel(r'$\rho\,/\,\mathrm{g}\,\mathrm{cm}^{-3}$')
   ax2[0].set_ylabel(r'Specific entropy $s\,/10^9\,\mathrm{erg}\,\mathrm{g}^{-1}\,\mathrm{K}^{-1}$')
   ax2[1].set_ylabel(r'$T\,/\,\mathrm{K}$')
   ax3[0].set_ylabel(r'$|\rho \mathbf{g}|$, $|\nabla P|\,/\,\mathrm{dyn}\,\mathrm{cm}^{-3}$')
   ax3[1].set_ylabel(r'$|\rho \mathbf{g} - \nabla P|\,/\,\mathrm{max}(|\rho \mathbf{g}|,|\nabla P|)$')

   if range is not None:
      ax1[0].set_xlim(range[0]/rsol, range[1]/rsol)
      ax2[0].set_xlim(range[0]/rsol, range[1]/rsol)
   
   ax2[0].set_ylim(entr.min()/1.1, entr.max()*1.1)

   ax1[0].legend(frameon=False)
   ax1[1].legend(frameon=False)
   ax2[0].legend(frameon=False)
   ax2[1].legend(frameon=False)
   ax3[0].legend(frameon=False, ncol=2)
   ax3[1].legend(frameon=False)

   fig1.tight_layout()
   fig2.tight_layout()
   fig3.tight_layout()
   fig1.savefig(basename+'-Mach-rho.pdf', format='pdf', bbox_inches='tight')
   fig2.savefig(basename+'-s-T.pdf', format='pdf', bbox_inches='tight')
   fig3.savefig(basename+'-hydro-eq.pdf', format='pdf', bbox_inches='tight')


def compute_radius(s):
   center_index = np.argmax(s.rho)
   center = s.pos[center_index].flatten()
   s.radii = s.r(center)


def compute_velocity(s):
   s.data['abs_vel'] = np.sqrt((s.vel**2).sum(axis=1))


def eos_results(props):
   # this has to be defined at the top-level 
   # because otherwise we cannot pickle it for use with multiprocessing
   res = eos.tgiven(props[0], props[1], props[2])
   return [res['s'], res['csnd']]


def compute_eos_stuff(s, number_processes=2):
   sound = np.zeros_like(s.rho)

   pool = mp.Pool(processes=number_processes)
   res = pool.map(eos_results, [(rho, temp, xh) for rho, temp, xh in zip(s.rho, s.temp, s.xnuc00)])
   res = np.asarray(res, dtype=np.float64).T

   if not 'entr' in s.data:
      s.data['entr'] = res[0]
   s.data['csnd'] = res[1]

   # free memory
   pool.close()
   pool.join()


def compute_accelerations(s):
   grad_p = s.data['grap'] # pressure gradient grad(P)
   grav_acc = s.data['rho'][...,np.newaxis]*s.data['acce'] # rho*vec(a)
   #acc_tot = s.data['rho'][...,np.newaxis]*s.data['acce'] # rho*vec(a)
   #grav_acc = acc_tot - grad_p
   #abs_acc_tot = np.sqrt(((acc_tot)**2).sum(axis=1))
   abs_grav_acc = np.sqrt(((grav_acc)**2).sum(axis=1))
   abs_grad_p = np.sqrt((grad_p**2).sum(axis=1))
   abs_diff = np.sqrt(((grav_acc - grad_p)**2).sum(axis=1))

   max_abs_a = np.maximum(abs_grav_acc, abs_grad_p)
   rel_dacc_norm = abs_diff/max_abs_a

   s.data['abs_grav_acc'] = abs_grav_acc
   s.data['abs_grad_p'] = abs_grad_p
   s.data['rel_dacc_norm'] = rel_dacc_norm


def set_isotope_data(s):
   for k, iso in enumerate(isotopes):
      s.data[iso] = s.data['xnuc'][:,k]


def compute_1d_quantities(s, qnames=['rho', 'pres'], range=None, nbins=None):
   quantities = {}
   for q in qnames:
      quantities[q] = get_averaged_1d_quantity(
                        s.radii, s.data[q], s.mass, 
                        range=range, nbins=nbins)
   quantities['mass'] = get_1d_quantity(  
                           s.radii, s.data['mass'], 
                           range=range, nbins=nbins, normalize=False)
   quantities['volume'] = get_1d_quantity(
                           s.radii, s.data['vol'], 
                           range=range, nbins=nbins, normalize=False)
   return quantities


def get_1d_quantity(radii, values, range=None, normalize=True, nbins=None):
   weights = np.ones_like(radii).astype(np.float64)
   return get_averaged_1d_quantity(radii, values, weights, range=range, normalize=normalize, nbins=nbins)


def get_averaged_1d_quantity(radii, values, weights, range=None, normalize=True, nbins=None):
   # throw away zero radii because of log function
   nonzero = radii > 0.0
   radii = radii[nonzero]
   values = values[nonzero]
   weights = weights[nonzero]
   # determine range
   if range is None:
      minimum_r = radii.min()
      maximum_r = radii.max()
      range = [minimum_r, maximum_r]
   range = np.log10(np.asarray(range))
   # determine number of bins (-> Freedman-Diaconis rule)
   if nbins is None:
      nbins = np.ceil(np.power(len(radii), 1./3.)).astype(np.int64)

   radii_log = np.log10(radii)
   norm, edges = np.histogram(radii_log, range=range, bins=nbins, weights=weights)
   quantity, edges = np.histogram(
         radii_log, range=range, bins=nbins,
         weights=values.astype(np.float64)*weights)
   nonempty_bins = norm > 0
   if normalize:
      quant_normalized = quantity[nonempty_bins]/norm[nonempty_bins]
   else:
      quant_normalized = quantity[nonempty_bins]

   # throw away empty bins
   last_non_empty_bin = np.arange(len(nonempty_bins))[nonempty_bins][-1]
   left_edges = edges[:-1]
   right_edges = edges[1:]
   edges = np.r_[left_edges[nonempty_bins], right_edges[last_non_empty_bin]]
   midpoints = np.power(10, 0.5*(edges[1:] + edges[:-1]))

   data = {'radius': midpoints, 'values': quant_normalized, 'edges': np.power(10, edges)}
   return data


def key_exists(arr, key):
   ret = False
   try:
      t = arr[key]
      ret = True
   except:
      pass
   return ret



if __name__ == "__main__":
   check_relaxation(models, output_path, number_processes)
