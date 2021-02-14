import numpy
import matplotlib.pyplot as plt
from numba import jit
import math

apply_nopython = True
apply_fastmath = True

#---------------------------------------------------------#
#constants
pi      = numpy.pi
sigma   = 1./(pi**1.5)

#indices to access parameters
i_N     = 0
i_h     = 1
i_gamma = 2
i_K     = 3
i_mp    = 4
i_G     = 5
i_nu    = 6
i_eps   = 7
i_CFL   = 8
noff    = 1
#---------------------------------------------------------#
'''
functions that get the total acceleration for each particle
a = a(pressure gradient) + a(gravity) + a(damping)
the time step estimate for the next time step is calculated in this functions
to optimize operations
INPUT:
x: particle positions (3,N)
v: particle velocities (3,N)
m: particle masses (N)
rho: density array (N) (implicitly returned in output, meaning that the program will
know about changes applied on this variable even if it is not explicitly returned by the function)
P: pressure array (N) (implicitly returned in output)
a: acceleration array (N)
pars: array with the input parameters
OUTPUT:
dt: time step estimate for the next leapfrog cycle
'''
@jit(nopython=apply_nopython,fastmath=apply_fastmath)
def get_acceleration(x,v,m,rho,P,a,pars):

    '''
    we get the time step estimate by getting the minimum between
    dt and the CFL time step for each particle
    '''
    dt = 10000.

    #TODO: calculate energy conservation
    EKIN  = 0.
    EGRAV = 0.
    EINT  = 0.

    #get dx,rho,P for each particle
    for i in range(0,pars[i_N]):

        rho[i] = 0. #initialize the density to 0

        '''
        there is no need to iterate a second time over the full array shape
        since the kernel is symmetric: ij = ji
        '''
        for j in range(0,i+noff):

            dx = x[0,i]-x[0,j]
            dy = x[1,i]-x[1,j]
            dz = x[2,i]-x[2,j]
            dr = (dx**2+dy**2+dz**2)**0.5

            #calculate the density using the formula you saw in class
            rho[i] += m[j]*sigma/(pars[i_h]**3)*math.exp(-(dr)**2/(pars[i_h])**2)

            #to avoid double counting
            if i!=j:
                 rho[j] += m[i]*sigma/(pars[i_h]**3)*math.exp(-(dr)**2/(pars[i_h])**2)



    #get a= pressure + gravity + damping term
    for i in range(0,pars[i_N]):

        #polytropic EOS
        P[i]   = pars[i_K]*rho[i]**pars[i_gamma]

        #TODO:
        EKIN += 0.5*m[i]*(v[0,i]**2+v[1,i]**2+v[2,i]**2)
        EINT += m[i]*P[i]/rho[i]/(pars[i_gamma]-1.)

        #time step estimate
        dt = min(dt,pars[i_CFL]*pars[i_h]/(pars[i_gamma]*P[i]/rho[i])**0.5)

        #initialize acceleration
        a[0,i] = 0.
        a[1,i] = 0.
        a[2,i] = 0.

        for j in range(0,i+noff):

            dx = x[0,i]-x[0,j]
            dy = x[1,i]-x[1,j]
            dz = x[2,i]-x[2,j]
            dr = (dx**2+dy**2+dz**2)**0.5

            #kernel derivatives, antisymmetric for ij -> ji.
            Wx = -2.*dx*sigma/(pars[i_h]**5)*math.exp(-(dr)**2/(pars[i_h])**2)
            Wy = -2.*dy*sigma/(pars[i_h]**5)*math.exp(-(dr)**2/(pars[i_h])**2)
            Wz = -2.*dz*sigma/(pars[i_h]**5)*math.exp(-(dr)**2/(pars[i_h])**2)


            a[0,i] += -m[j]*(P[i]/rho[i]**2+P[j]/rho[j]**2)*Wx \
            -pars[i_G]*m[j]/(dr**3+pars[i_eps]**3)*dx

            a[1,i] += -m[j]*(P[i]/rho[i]**2+P[j]/rho[j]**2)*Wy \
            -pars[i_G]*m[j]/(dr**3+pars[i_eps]**3)*dy

            a[2,i] += -m[j]*(P[i]/rho[i]**2+P[j]/rho[j]**2)*Wz \
            -pars[i_G]*m[j]/(dr**3+pars[i_eps]**3)*dz


            #again to void double counting
            if i!=j:

                a[0,j] -= -m[i]*(P[i]/rho[i]**2+P[j]/rho[j]**2)*Wx \
                -pars[i_G]*m[j]/(dr**3+pars[i_eps]**3)*dx

                a[1,j] -= -m[i]*(P[i]/rho[i]**2+P[j]/rho[j]**2)*Wy \
                -pars[i_G]*m[j]/(dr**3+pars[i_eps]**3)*dy

                a[2,j] -= -m[i]*(P[i]/rho[i]**2+P[j]/rho[j]**2)*Wz \
                -pars[i_G]*m[j]/(dr**3+pars[i_eps]**3)*dz

                #TODO
                EGRAV+=-pars[i_G]*m[j]*m[i]/dr


        #add damping term
        a[0,i] -= pars[i_nu]*v[0,i]
        a[1,i] -= pars[i_nu]*v[1,i]
        a[2,i] -= pars[i_nu]*v[2,i]

    ETOT = EKIN+EINT+EGRAV 
    return dt,ETOT

#---------------------------------------------------------#
'''
functions that updates the x and v particles for each particle using the leapfrog
algorithm
INPUT:
x: particle positions (3,N)
v: particle velocities (3,N)
m: particle masses (N)
rho: density array (N) (implicitly returned in output, meaning that the program will
know about changes applied on this variable even if it is not explicitly returned by the function)
P: pressure array (N) (implicitly returned in output)
a: acceleration array (N)
pars: array with the input parameters (N)
dt: time step estimate from previous leapfrog cycle
OUTPUT:
dt: time step estimate for the next leapfrog cycle calculated at the middle step in get_acceleration
'''
@jit(nopython=apply_nopython,fastmath=apply_fastmath)
def leapfrog(x,v,m,rho,P,a,dt,pars):

    #---------------------------------------------------#
    '''
    1) v(dt/2)=v(t)+a dt/2
    2) x(t+dt)=x(t)+v(dt/2)*dt
    '''
    for i in range(0,pars[i_N]):

        v[0,i] += a[0,i]*dt/2.
        v[1,i] += a[1,i]*dt/2.
        v[2,i] += a[2,i]*dt/2.

        x[0,i] += v[0,i]*dt
        x[1,i] += v[1,i]*dt
        x[2,i] += v[2,i]*dt

    '''
    3) get a(t+dt)
    '''
    #---------------------------------------------------#
    dtnew,E=get_acceleration(x,v,m,rho,P,a,pars)
    #---------------------------------------------------#



    '''
    4) v(t+dt)=v(t+dt/2)+a(t+dt)*dt/2
    '''
    for i in range(0,N):

        v[0,i] += a[0,i]*dt/2.
        v[1,i] += a[1,i]*dt/2.
        v[2,i] += a[2,i]*dt/2.
    #---------------------------------------------------#

    return dtnew,E


#---------------------------------------------------------#

'''
MAIN PROGRAM:
setup the ICs and run
'''
#------------------------#



numpy.random.seed(541) #seed for random number genereation

N     =       800      #number of particles
h     =       0.1      #kernel size
gamma =       2        #polytropic index
K     =       0.1      #pressure constant: P=K rho**gamma
mp    =       0.0025   #particle mass
G     =       0.1      #rescaled gravitational constant
nu    =       0.       #damping coefficient
eps   =       h/100    #softening length
CFL   =       0.8      #CFL factor for getting sonic time step


pars = numpy.array((N,h,gamma,K,mp,G,nu,eps,CFL)) #define array with parameters

#------------------------#
'''
variable definitions
0: -> x component
1: -> y component
2: -> z component
'''
x     = numpy.zeros((3,N))
v     = numpy.zeros((3,N))
a     = numpy.zeros((3,N))
m     = numpy.zeros((N))
rho   = numpy.zeros((N))
P     = numpy.zeros((N))

#------------------------#
#setting initial conditions

data = numpy.load('numpy.npz')

x[0,:N//2] = data['x'][0]
x[1,:N//2] = data['x'][1]
x[2,:N//2] = data['x'][2]

v[0,:N//2] = data['v'][0]
v[1,:N//2] = data['v'][1]
v[2,:N//2] = data['v'][2]

x[0,N//2:] = data['x'][0] + 3.
x[1,N//2:] = data['x'][1] + 1.5
x[2,N//2:] = data['x'][2] 

v[0,N//2:] = data['v'][0] - 0.3
v[1,N//2:] = data['v'][1]
v[2,N//2:] = data['v'][2]




#set the masses
m[:] = mp

#------------------------#
#setup figure object with axis
grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.3)
ax1  = plt.subplot(grid[0])

'''
get dt and acceleration before entering the time loop. This is needed for the
leapfrog algorithm
'''
dt,E0=get_acceleration(x,v,m,rho,P,a,pars)



#main time loop
t    = 0.0    #initial time
tmax = 30.    #maximum time
z=0
while(t<tmax):


    t+=dt #update time
    dt,E=leapfrog(x,v,m,rho,P,a,dt,pars) #leapfrog step and next time step estimate

    #plotting instructions:
    plt.sca(ax1)
    plt.cla()
    plt.scatter(x[0,:],x[1,:],color='blue',s=10, alpha=0.5)
    ax1.set(xlim=(-3,3), ylim=(-3,3))
    ax1.set_aspect('equal', 'box')
    ax1.set_title('t=%.3f   '%(t)+r'$\frac{E}{E_0}=%.5f$'%(E/E0))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.pause(0.001)
    plt.savefig('numba_%d.png'%(z))
    z+=1


#------------------------#
