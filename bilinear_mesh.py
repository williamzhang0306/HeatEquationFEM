'''

'''

import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Callable

# basis functions for bilienar parent element
# defined over the interval [-1,1]
phi1 = lambda chi: (1 - chi)/2
phi2 = lambda chi: (chi + 1)/2
dphi1_dchi = -1/2
dphi2_dchi = 1/2
phis = [phi1,phi2]
dphi_dchis = [-1/2, 1/2]

def quadrature(f: Callable, order:int = 2):
    '''
    Integrates f(x)dx over the interval x in [-1,1]
    '''
    if order not in [2,3]:
        raise ValueError("Quadrature order must be integer 2 or 3.")

    if order == 2:
        points = [-1/math.sqrt(3), 1/math.sqrt(3)]
        weights = [1, 1]
        
    elif order == 3:
        points = [-math.sqrt(3/5), 0 , math.sqrt(3/5)]
        weights = [5/9, 8/9, 5/9]

    #print([(xi,wi) for (xi,wi) in zip(points,weights)])
    return sum([wi * f(xi) for (xi,wi) in zip(points,weights)])


class BilinearUniformMesh():
    '''
    Helper functions for solving  1D, time dependent PDEs on the
    interval x in [0,1] using the Galerkin FEM with uniform, bilinear elements.
    '''

    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.n_elements = n_nodes - 1
        self.connectivty = [(x,x+1) for x in range(0,n_nodes)]
        self.h = 1/self.n_elements # mesh spacing
        self.dx_dchi = self.h/2 # jacobian mapping



def get_stiffness_matrix(n_nodes, connectivity, dx_dchi, dchi_dx):
    
    # global K
    K = np.zeros([n_nodes,n_nodes])

    for element in range(0,n_nodes-1):
        # calculate the local k matrix
        # If the mesh is not uniform this must be recalculated per element
        klocal = np.zeros([2,2])

        for i in range(0,2):
            for j in range(0,2):
                integrand = lambda chi: \
                    dchi_dx*dphi_dchis[i] * dchi_dx * dphi_dchis[j] * dx_dchi

                klocal[i,j] = quadrature(integrand) # integrate with respect to chi

        # map local k matrix to global K matrix
        for l in range(0,2):
            for m in range(0,2):
                global_node_1 = connectivity[element][l]
                global_node_2 = connectivity[element][m]
                K[global_node_1][global_node_2] += klocal[l][m]

    return K

def get_mass_matrix(n_nodes, connectivity, dx_dchi, dchi_dx):
    
    # global K
    M = np.zeros([n_nodes,n_nodes])

    for element in range(0,n_nodes-1):
        # calculate the local k matrix
        # If the mesh is not uniform this must be recalculated per element
        mlocal = np.zeros([2,2])

        for i in range(0,2):
            for j in range(0,2):
                integrand = lambda chi: \
                    2 * phis[i](chi) * phis[j](chi) * dx_dchi

                mlocal[i,j] = quadrature(integrand, order =2) # integrate with respect to chi

        print(mlocal)

        # map local k matrix to global K matrix
        for l in range(0,2):
            for m in range(0,2):
                global_node_1 = connectivity[element][l]
                global_node_2 = connectivity[element][m]
                M[global_node_1][global_node_2] += mlocal[l][m]

    return M

def get_force_matrix(n_nodes,connectivity, dx_dchi, f, t_elapsed, h):
    
    F = np.zeros(n_nodes)

    for k in range(0,n_nodes-1):
        # calculate local f (on the boundaries)
        flocal = [0, 0]
        for l in range(0,2):
            xi = k * h
            integrand = lambda chi: \
                f((chi+1)*h/2 + xi, t_elapsed)*phis[l](chi)*dx_dchi
            flocal[l] = quadrature(integrand, order = 3)
        
        print(flocal)
        # map f local to f global
        for l in range(0,2):
            global_node = connectivity[k][l]
            F[global_node] += flocal[l]

    return F


def test():
    n_nodes = 11
    n_elements = n_nodes -1 
    connectivity = [(x,x+1) for x in range(0,n_elements)]
    h = 1/n_elements
    dx_dchi = h/2
    dchi_dx = 2/h

    K = get_stiffness_matrix(n_nodes, connectivity, dx_dchi, dchi_dx)
    M = get_mass_matrix(n_nodes, connectivity, dx_dchi, dchi_dx)

    f = lambda x,t : (math.pi**2 - 1)*math.exp(-t)*math.sin(math.pi*x)
    F = get_force_matrix(n_nodes, connectivity, dx_dchi, f,t_elapsed=1/551, h=h)
    print(F)

