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
N = 1000
print(f"N={N}")
a = 30e-10


#V = 0.5*x**2
D = np.linspace(0, 3e-10, 10)



m = electron_mass
Lambda = hbar**2 / (2 * m)

prob = []
for d in D: 
    x = np.linspace(-a, 3*a+d, N+1) # define our grid

    dx = x[1]-x[0]
    #print(f"dx = {dx}")
    d_dx = FinDiff(0, dx, 1)
    d2_dx2 = FinDiff(0, dx, 2)
    print(d)
    V = x*0
    V[x < 0] = 3*e
    V[x > a] = 3*e
    V[x>a + d] = 0
    V[x>a +d +a] =3*e

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
    normalisation = np.sum(states[0].real**2)
    states = states / normalisation
    print(len(x), len(states[0]))
    tunnel = states[0][x>a]
    x2 = x[x>a]
    tunnel = tunnel[x2<a+d]
    prob.append(np.sum(tunnel.real**2))

    #print(energies[0:5])
    print(a)
    print("E:")
    print(energies[0:3])
    print(energies[1]-energies[0], Planck * c/10.6e-6)
    #exit()
    # fig = plt.figure()

    # plt.plot(x, V*0.1e17)
    # plt.plot(x, states[0].real**2, label=r'$\psi_1$')
    
    # plt.plot(x, states[2].real**2, label=r'$\psi_3$')
    # plt.plot(x, states[5].real**2, label=r'$\psi_3$')
    
    # #plt.plot(x, states[3].real, label=r'$\psi_4$')
    # #plt.plot(x, states[4].real, label=r'$\psi_5$')
    # #plt.plot(x,states[50].real**2, label=r'$\psi_{60}$') # Compare to Figure 2.7 in Griffith
    # plt.grid()
    # #plt.tick_params('both', direction = "in")
    # plt.legend()
    # plt.xlim(min(x), max(x))
    # plt.show()
    #fig.savefig("puit infini.pdf")

print(f"Prob: {prob}")
plt.plot(D, prob, 'k')
plt.xlabel("d (m)")
plt.ylabel("Probabilité (m^-2)")
plt.show()
exit()

fig = plt.figure(figsize=(5,8))
ax = fig.gca()
levels = [[(0, 1), (e.real, e.real)] for e in energies]
for level in levels[:5]:    
    ax.plot(level[0], level[1], '-b')
ax.set_xticks([])
ax.set_ylabel('energy [a.u.]')
plt.show()
