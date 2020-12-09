import numpy as np
import matplotlib.pyplot as plt
import os

# Fundamentals of Simulation Methods
# Problem Set 5
# Author: Elias Olofsson
# Date: 2020-12-09

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]

def analyse_data(temperature):

        txtname = 'output_T' + str(temperature) + '.txt'
        
        # Import and split up.
        data = np.loadtxt(txtname)
        t = data[:,0]
        K = data[:,1]
        V = data[:,2]
        E = data[:,3]

        # Plot the energies.
        plt.plot(t, np.abs(K), label='Kinetic')
        plt.plot(t, np.abs(V), label='Potential')
        plt.plot(t, np.abs(E), label='Total')
        plt.yscale('log')
        plt.xlabel('Time [1]')
        plt.ylabel('Energy [1]')
        plt.title(f'Energy evolution, initial temperature T = {temperature}K')
        plt.legend()
        plt.savefig(f'fig_energy{temperature}.pdf')
        

        # Plot relative error of total energy.
        plt.figure()
        plt.plot(t, np.abs((E-E[0])/E[0]), label=f'T = {temperature}K')
        plt.yscale('log')
        plt.xlabel('Time [1]')
        plt.ylabel('(E-E[0])/E[0] [1]')
        plt.title(f'Relative energy error, intial temperature T = {temperature}K')
        plt.legend()
        plt.savefig(f'fig_err{temperature}.pdf')

        # Auto-correlation of absolute value of velocity.
        v = np.sqrt(2*K)
        v_autocorr = autocorr(v)

        plt.figure()
        plt.plot(v_autocorr/np.max(v_autocorr), label='Autocorrelation abs(v)')
        plt.hlines(1/np.exp(1), 0, 2000, label='1/e')
        plt.xlabel('Delay in time')
        plt.ylabel('Relative Overlap')
        plt.title(f'Autocorrelation of average absolute velocity, T = {temperature}K')
        plt.legend()
        plt.savefig(f'fig_auto{temperature}.pdf')

        # Kinetic temperature
        plt.figure()
        plt.plot(t, K*2/3*120)
        plt.xlabel('Time [1]')
        plt.ylabel('Temperature [K]')
        plt.title(f'Kinetic temperature, initial temperaure T = {temperature}K')
        plt.savefig(f'fig_temp{temperature}.pdf')



analyse_data(80)
analyse_data(400)


