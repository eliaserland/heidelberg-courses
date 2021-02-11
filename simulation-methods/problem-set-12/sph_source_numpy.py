import numpy
import matplotlib.pyplot as plt
import math


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
axs     = 1




def W(dr,pars):
    return sigma/(pars[i_h]**3)*numpy.exp(-(dr)**2/(pars[i_h])**2)

def dW_dxi(dxi,dr,pars):
    return  -2.*dxi*W(dr,pars)/(pars[i_h])**2


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
def get_acceleration(x,v,m,rho,P,a,pars):

    '''
    we get the time step estimate by getting the minimum between
    dt and the CFL time step for each particle
    '''
    dt = 10000.



    #get dx,rho,P for each particle
    xi = numpy.reshape(x[0,:],(int(pars[i_N]),1))
    yi = numpy.reshape(x[1,:],(int(pars[i_N]),1))
    zi = numpy.reshape(x[2,:],(int(pars[i_N]),1))

    xj = numpy.reshape(x[0,:],(int(pars[i_N]),1)).T
    yj = numpy.reshape(x[1,:],(int(pars[i_N]),1)).T
    zj = numpy.reshape(x[2,:],(int(pars[i_N]),1)).T

    dx = xi-xj
    dy = yi-yj
    dz = zi-zj

    dr          = (dx**2+dy**2+dz**2)**0.5
    rho[:]      = numpy.sum(m[:]*W(dr,pars),axs)
    P[:]        = pars[i_K]*(rho**pars[i_gamma])


    EKIN = numpy.sum(0.5*m[:]*(v[0,:]**2+v[1,:]**2+v[2,:]**2))
    EINT = numpy.sum(m[:]*P[:]/rho[:]/(pars[i_gamma]-1))

    cs           = (pars[i_gamma]*P/rho)**0.5
    dt           = pars[i_CFL]*pars[i_h]/cs.max()

    Wx = dW_dxi(dx,dr,pars)
    Wy = dW_dxi(dy,dr,pars)
    Wz = dW_dxi(dz,dr,pars)


    #get a= pressure + gravity + damping term
    Pa   = numpy.reshape(P,(int(pars[i_N]),1))
    Pb   = numpy.reshape(P,(int(pars[i_N]),1)).T
    rhoa = numpy.reshape(rho,(int(pars[i_N]),1))
    rhob = numpy.reshape(rho,(int(pars[i_N]),1)).T



    a[0,:] = -numpy.sum( m[:]*((Pa/rhoa**2)+(Pb/rhob**2))*Wx , axs)
    a[1,:] = -numpy.sum( m[:]*((Pa/rhoa**2)+(Pb/rhob**2))*Wy , axs)
    a[2,:] = -numpy.sum( m[:]*((Pa/rhoa**2)+(Pb/rhob**2))*Wz , axs)


    mj = numpy.reshape(m,(int(pars[i_N]),1)).T
    a[0,:]  += -numpy.sum(pars[i_G]*mj/(dr**3+pars[i_eps]**3)*dx, axs)
    a[1,:]  += -numpy.sum(pars[i_G]*mj/(dr**3+pars[i_eps]**3)*dy, axs)
    a[2,:]  += -numpy.sum(pars[i_G]*mj/(dr**3+pars[i_eps]**3)*dz, axs)

    a[0,:] -= pars[i_nu]*v[0,:]
    a[1,:] -= pars[i_nu]*v[1,:]
    a[2,:] -= pars[i_nu]*v[2,:]


    dr[dr==0] = 1e+30
    EGRAV = numpy.sum(-pars[i_G]*mj*m[:]/dr,axs)
    EGRAV = 0.5*numpy.sum(EGRAV)

    ETOT = EKIN+EINT+EGRAV 
    return dt,ETOT


def leapfrog(x,v,m,rho,P,a,dt,pars):

    #---------------------------------------------------#
    '''
    1) v(dt/2)=v(t)+a dt/2
    2) x(t+dt)=x(t)+v(dt/2)*dt
    '''
    v[0,:] = v[0,:] + a[0,:]*dt/2.
    v[1,:] = v[1,:] + a[1,:]*dt/2.
    v[2,:] = v[2,:] + a[2,:]*dt/2.

    x[:,:] += v[:,:]*dt


    '''
    3) get a(t+dt)
    '''
    #---------------------------------------------------#
    dtnew,ETOT=get_acceleration(x,v,m,rho,P,a,pars)
    #---------------------------------------------------#

    '''
    4) v(t+dt)=v(t+dt/2)+a(t+dt)*dt/2
    '''

    v[0,:] = v[0,:] + a[0,:]*dt/2.
    v[1,:] = v[1,:] + a[1,:]*dt/2.
    v[2,:] = v[2,:] + a[2,:]*dt/2.
    #---------------------------------------------------#

    return dtnew,ETOT






#------------------------------------------_#

'''
MAIN PROGRAM:
setup the ICs and run
'''
#------------------------#



numpy.random.seed(541) #seed for random number genereation

N     =       1600 #number of particles
h     =       0.1#kernel size
gamma =       2.#polytropic index
K     =       0.1#pressure constant: P=K rho**gamma
mp    =       0.0025#particle mass
G     =       0.1#rescaled gravitational constant
nu    =       0.#damping coefficient
eps   =       h/100.#softening length
CFL   =       0.8#CFL factor for getting sonic time step


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
t    = 0.0 #initial time
tmax = 30  #maximum time
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
    plt.savefig('numpy_%d.png'%(z))
    plt.pause(0.001)
    z+=1

#------------------------#


