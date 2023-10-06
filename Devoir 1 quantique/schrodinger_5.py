import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from findiff import FinDiff
import matplotlib.pyplot as plt
from scipy.constants import hbar, e, electron_mass, Planck, c
""" 
Reference
https://medium.com/@mathcube7/two-lines-of-python-to-solve-the-schrödinger-equation-2bced55c2a0e
"""
N = 20000
print(f"N={N}")
a = 27.014e-10
x = np.linspace(-a, 2*a, N+1) # define our grid

dx = x[1]-x[0]
#print(f"dx = {dx}")
d_dx = FinDiff(0, dx, 1)
d2_dx2 = FinDiff(0, dx, 2)

#V = 0.5*x**2
V = x*0
V[x < 0] = e
V[x > a] = e

m = electron_mass
Lambda = hbar**2 / (2 * m)
H = -Lambda*d2_dx2.matrix(x.shape) + diags(V)

# Matrix = H
# print(type(Matrix))
# print((Matrix))
# d=Matrix.todense()
# plt.imshow(d,interpolation='none',cmap='binary',extent=[-2, 2, -2, 2])
# plt.colorbar()
# ax = plt.gca()
# ax.set_xlabel([-2,2])
# plt.show()
# exit()
energies, states = eigs( H, k=60, which="SR", v0=x*0+1)#, which='SR')

sortE = sorted(zip(energies,states.T),\
                                    key=lambda x: x[0].real, reverse=False)
sortE = np.array(sortE, dtype = 'object')
energies = []
states = []
print("ma^2/hbar^2 * E:")
for i in sortE:
    energies.append(i[0].real)
    states.append(i[1])

energies = np.array(energies)
states = np.array(states)
#print(energies[0:5])
print(a)
print("E:")
print(energies[0:3])
print(energies[1]-energies[0], Planck * c/10.6e-6)
#exit()
fig = plt.figure()
plt.plot(x, states[0].real, label=r'$\psi_1$')
plt.plot(x, states[1].real, label=r'$\psi_2$')
plt.plot(x, states[2].real, label=r'$\psi_3$')
#plt.plot(x, states[3].real, label=r'$\psi_4$')
#plt.plot(x, states[4].real, label=r'$\psi_5$')
#plt.plot(x,states[50].real**2, label=r'$\psi_{60}$') # Compare to Figure 2.7 in Griffith
plt.grid()
#plt.tick_params('both', direction = "in")
plt.legend()
plt.xlim(min(x), max(x))
plt.show()
#fig.savefig("puit infini.pdf")
exit()

fig = plt.figure(figsize=(5,8))
ax = fig.gca()
levels = [[(0, 1), (e.real, e.real)] for e in energies]
for level in levels[:5]:    
    ax.plot(level[0], level[1], '-b')
ax.set_xticks([])
ax.set_ylabel('energy [a.u.]')
plt.show()
