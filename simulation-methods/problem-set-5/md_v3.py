import numpy
from matplotlib.pyplot import *
import math
import time


#global parameters
pi = math.pi
i_x = 0 #index to get x-coord
i_y = 1 #index to get y-coord
i_z = 2 #index to get z-coord
i_u = 3 #index to get vx
i_v = 4 #index to get vy
i_w = 5 #index to get vz



def rand_gauss():

           #---students---#
           #TODO 1.b: write a function that returns one random number sampled from a normal distribution (mean=0 and std=1)


           #---end---#



#this function returns the distance from every other particle for particle i: x-y-z components
def dx_vec(i,p):

          dx = p.coord[i,i_x]-p.coord[:,i_x]
          dy = p.coord[i,i_y]-p.coord[:,i_y]
          dz = p.coord[i,i_z]-p.coord[:,i_z]

          kx1 = numpy.where((numpy.abs(dx)>(p.L/2.)) & (dx<0.))
          kx2 = numpy.where((numpy.abs(dx)>(p.L/2.)) & (dx>0.))

          ky1 = numpy.where((numpy.abs(dy)>(p.L/2.)) & (dy<0.))
          ky2 = numpy.where((numpy.abs(dy)>(p.L/2.)) & (dy>0.))

          kz1 = numpy.where((numpy.abs(dz)>(p.L/2.)) & (dz<0.))
          kz2 = numpy.where((numpy.abs(dz)>(p.L/2.)) & (dz>0.))

          dx[kx1] = p.L + dx[kx1] 
          dx[kx2] = dx[kx2] - p.L

          dy[ky1] = L + dy[ky1] 
          dy[ky2] = dy[ky2] - p.L

          dz[kz1] = L + dz[kz1] 
          dz[kz2] = dz[kz2] - p.L

          return dx,dy,dz
               



class particles:

     #input paramaters : 
     #L: size of the box 
     #N1d: number of particles per dimension
     #sig_v: thermal velocity (mean dispersion)
     #r_cut: cut radius
     #output_frequency: to store the output every "output_frequency" steps (it must be an integer)
     def __init__(self,L,N1d,sig_v,r_cut,output_frequency):

            self.N = N1d**3 #set total number of particles

            self.coord =  numpy.zeros((self.N,3)) #create structure for coordinates
            self.vel   =  numpy.zeros((self.N,3)) #same for the velocities

            dl = L/N1d #set spatial separation
            
            ip = 0

            #TODO 1.b: initialize coordinates and velocities
            for i in range(N1d):
                for j in range(N1d):
                      for k in range(N1d):

                               #---students---#

                               self.coord[ip,i_x] = 
                               self.coord[ip,i_y] = 
                               self.coord[ip,i_z] =  
                               self.vel[ip,i_x]   =  
                               self.vel[ip,i_y]   =  
                               self.vel[ip,i_z]   =  
                               ip+=1
                               
                               #---end---#
 

            self.F     =  numpy.zeros((self.N,3)) #force array
            self.a     =  numpy.zeros((self.N,3)) #acceleration array
            self.R     =  numpy.zeros((self.N,6)) #residual (to use in the time marching scheme)

            self.time         = 0. #initialize the time
            self.dt           = 0.01 #intialize the time step (dummy)
            self.L            = L #store the size of the box
            self.r_cut        = r_cut #same for cut radius
            self.model_number = 0 #initialize model number
            self.noutput      = 0 #initialize the number of outputs already stored
            self.E0           = 0.#total initial energy (dummy)




            self.output_frequency = output_frequency #initialize output frequency
            self.max_output       = 500 #maximum number of outputs


     #calculate the acceleration for each particle (we also get the potential energy to avoid doing the same calculation twice)
     def get_acceleration(self):

            self.V = 0. #initialize potential energy
            self.rmin = 1e+9
            
            #for each particle
            for i in range(self.N):

                      #get distance from all the other particles (self particle included)
                      dx,dy,dz = dx_vec(i,self)
 
                      #---students---#

                      r2 = (dx**2+dy**2+dz**2) #get radii
                      self.F[:,:] = 0.0 #initialize forces acting on particle i :F = 0
                      k_int = (r2>0.) & (r2<self.r_cut**2) #get only the particles inside cut radius and exclude self
                      r     = r2[k_int]**0.5
  
                      #TODO 1.c: calculate the force acting on particle i due to all the other particles
                      self.F[k_int,i_x] =  
                      self.F[k_int,i_y] =  
                      self.F[k_int,i_z] =  

                      self.a[i,:] = numpy.sum(self.F,axis=0)
                      #TODO: 2.b update the potential for particle i and add to the total potential energy 
                      V = 
                      self.V += 

                      #---end---#



     #evolve the coordinates according to residuals and apply periodic BCs
     def evolve_coords(self):

   
              kx1 = numpy.where((self.coord[:,i_x]+self.R[:,i_x])>self.L)
              kx2 = numpy.where((self.coord[:,i_x]+self.R[:,i_x])<0.)
              ky1 = numpy.where((self.coord[:,i_y]+self.R[:,i_y])>self.L)
              ky2 = numpy.where((self.coord[:,i_y]+self.R[:,i_y])<0.)
              kz1 = numpy.where((self.coord[:,i_z]+self.R[:,i_z])>self.L)
              kz2 = numpy.where((self.coord[:,i_z]+self.R[:,i_z])<0.)

              kx = numpy.where(((self.coord[:,i_x]+self.R[:,i_x])>0.) & ((self.coord[:,i_x]+self.R[:,i_x])<self.L))
              ky = numpy.where(((self.coord[:,i_y]+self.R[:,i_y])>0.) & ((self.coord[:,i_y]+self.R[:,i_y])<self.L))
              kz = numpy.where(((self.coord[:,i_z]+self.R[:,i_z])>0.) & ((self.coord[:,i_z]+self.R[:,i_z])<self.L))

              self.coord[kx1,i_x] = self.coord[kx1,i_x] + self.R[kx1,i_x] - self.L
              self.coord[ky1,i_y] = self.coord[ky1,i_y] + self.R[ky1,i_y] - self.L
              self.coord[kz1,i_z] = self.coord[kz1,i_z] + self.R[kz1,i_z] - self.L

              self.coord[kx2,i_x] = self.L + self.R[kx2,i_x] + self.coord[kx2,i_x]
              self.coord[ky2,i_y] = self.L + self.R[ky2,i_y] + self.coord[ky2,i_y] 
              self.coord[kz2,i_z] = self.L + self.R[kz2,i_z] + self.coord[kz2,i_z] 

              self.coord[kx,i_x] +=  self.R[kx,i_x] 
              self.coord[ky,i_y] +=  self.R[ky,i_y]  
              self.coord[kz,i_z] +=  self.R[kz,i_z]  
   




             
     #TODO 2.a: implement the leapfrog algorithm (follow the rk1 example given in the sheet)
     def do_leapfrog(self):


             #---students---#
            


             #---end---#
              
     #TODO: 2.b: get the total kinetic energy 
     def getK(self):

              #---students---#

              self.K = 

              #---end---#

     #TODO 2.b: get total energy
     def getE(self):


              self.getK()

              #---students---#

               self.E = 

              #---end---#
             
             
              if self.model_number == 0:
                  self.E0 = self.E

     #routines to store the output:
     #it stores the coordinates, the velocities, the kinetic energy, the potential energy, the total energy, the time and the number of particles
     
     def check_for_store(self):
        
              if (self.model_number%self.output_frequency==0):

                      self.store()
                  
                      
     def store(self):
             
              numpy.savez('%d.npz'%(self.noutput),coord=self.coord,vel=self.vel,K=self.K,V=self.V,E=self.E,t=self.time,Np=self.N)

              self.noutput += 1
 
              if self.noutput >= self.max_output:
                     self.output_frequency = 20000000000


     #prints some useful info on terminal to check the status of the simulation
     def print_info(self):

              print('| model number=%d | (E-E0)/E0: %.7e |'%(self.model_number,(self.E-self.E0)/self.E0))


   

#//////////////////////////////////////////////////////////////////////

#input parameters
#TODO 1.b setup the system
#---students---#
T                = 
sig_v            = 
N1d              = 
L                = 
r_cut            = 
output_frequency = 


Niter            = 
dt               = 

#---end---#

#main program 
t1 = time.time()

p = particles(L,N1d,sig_v,r_cut,output_frequency)
p.dt = dt




#---students---#
#TODO: #you have to complete the program to get the energy for every step and initialize the acceleration for the leapfrog


for tt in range(Niter):

    p.print_info()
    p.check_for_store()
    p.do_leapfrog()
    p.model_number+=1
    p.time += p.dt

t2 = time.time()
print(t2-t1)


#---end---#

#//////////////////////////////////////////////////////////////////////

