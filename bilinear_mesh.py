'''

'''

import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Callable


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

    return sum([wi * f(xi) for (xi,wi) in zip(points,weights)])


class BilinearUniformMesh():
    '''
    Helper functions for solving  1D, time dependent PDEs on the
    interval x in [0,1] using the Galerkin FEM with uniform, bilinear elements.
    '''

    def __init__(self, n_nodes):
        # uniform mesh properties
        self.n_nodes = n_nodes
        self.n_elements = n_nodes - 1
        self.h = 1/self.n_elements # node spacing

        # used to map local element node index to global node index
        self.connectivity = [(x,x+1) for x in range(0,self.n_nodes)] 
        
        # basis functions and first derivatives
        phi1 = lambda chi: (1 - chi)/2
        phi2 = lambda chi: (chi + 1)/2
        dphi1_dchi = -1/2
        dphi2_dchi = 1/2
        self.phis = [phi1,phi2]
        self.dphi_dchis = [dphi1_dchi, dphi2_dchi]

        # 1D Jacobians mapping x space to chi space
        self.dx_dchi = self.h/2 
        self.dchi_dx = 2/self.h 

    def get_node_locations(self) -> np.ndarray:
        '''
        Returns the spatial locations of the nodes.
        '''
        return np.array([i*self.h for i in range(0,self.n_nodes)])
    
    def get_stiffness_matrix(self):
        '''
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin sem 
        metus, viverra in lacus a, suscipit blandit enim. Morbi nisi velit, 
        feugiat quis scelerisque ac, sollicitudin ut ante. Mauris dictum dui 
        tellus, id tempus nisl tincidunt in. Mauris purus ante, interdum a pharetra 
        '''
        K = np.zeros([self.n_nodes,self.n_nodes])

        for element in range(0,self.n_elements):
            # calculate the local k matrix
            # If the mesh is not uniform this must be recalculated per element
            klocal = np.zeros([2,2])

            for i in range(0,2):
                for j in range(0,2):
                    integrand = lambda chi: self.dchi_dx*self.dphi_dchis[i] \
                                            * self.dchi_dx*self.dphi_dchis[j] \
                                            * self.dx_dchi

                    klocal[i,j] = quadrature(integrand) # integrate with respect to chi

            # map local k matrix to global K matrix
            for l in range(0,2):
                for m in range(0,2):
                    global_node_1 = self.connectivity[element][l]
                    global_node_2 = self.connectivity[element][m]
                    K[global_node_1][global_node_2] += klocal[l][m]

        return K
    
    def get_mass_matrix(self) -> np.ndarray:
        '''
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin sem 
        metus, viverra in lacus a, suscipit blandit enim. Morbi nisi velit, 
        feugiat quis scelerisque ac, sollicitudin ut ante. Mauris dictum dui 
        tellus, id tempus nisl tincidunt in. Mauris purus ante, interdum a pharetra 
        '''
        # global M
        M = np.zeros([self.n_nodes,self.n_nodes])

        for element in range(0,self.n_nodes-1):
            # calculate the local m matrix
            # If the mesh is not uniform this must be recalculated per element
            mlocal = np.zeros([2,2])

            for i in range(0,2):
                for j in range(0,2):
                    integrand = lambda chi: \
                        self.phis[i](chi) * self.phis[j](chi) * self.dx_dchi

                    mlocal[i,j] = quadrature(integrand, order =2) # integrate with respect to chi

            # map local k matrix to global K matrix
            for l in range(0,2):
                for m in range(0,2):
                    global_node_1 = self.connectivity[element][l]
                    global_node_2 = self.connectivity[element][m]
                    M[global_node_1][global_node_2] += mlocal[l][m]

        return M
    
    def get_force_matrix(self, f, t_elapsed)-> np.ndarray:
        '''
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin sem 
        metus, viverra in lacus a, suscipit blandit enim. Morbi nisi velit, 
        feugiat quis scelerisque ac, sollicitudin ut ante. Mauris dictum dui 
        tellus, id tempus nisl tincidunt in. Mauris purus ante, interdum a pharetra 
        '''
        F = np.zeros(self.n_nodes)

        for k in range(0,self.n_nodes-1):
            # calculate local f (on the boundaries)
            flocal = [0, 0]
            for l in range(0,2):
                xi = k * self.h
                integrand = lambda chi: \
                    f((chi+1)*self.h/2 + xi, t_elapsed)*self.phis[l](chi)*self.dx_dchi
                flocal[l] = quadrature(integrand, order = 3)
            
            # map f local to f global
            for l in range(0,2):
                global_node = self.connectivity[k][l]
                F[global_node] += flocal[l]

        return F


def test():
    N = 11
    mesh = BilinearUniformMesh(N)
    K = mesh.get_stiffness_matrix()
    M = mesh.get_mass_matrix()
    f = lambda x,t : (math.pi**2 - 1)*math.exp(-t)*math.sin(math.pi*x)
    F = mesh.get_force_matrix(f, 1/552)
    print(F)

if __name__ == '__main__':
    test()