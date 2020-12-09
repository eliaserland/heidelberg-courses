import numpy as np
import matplotlib.pyplot as plt
import os

data = np.loadtxt('output_T80.txt')

print(data.shape)

t = data[:,0]
K = data[:,1]
V = data[:,2]
E = data[:,3]

plt.plot(t, np.abs(K), label='Kinetic')
plt.plot(t, np.abs(V), label='Potential')
plt.plot(t, np.abs(E), label='Total')
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.show()


#plt.plot(t, V)
plt.show()




