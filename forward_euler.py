# un+1 = [I - dt Minv K] un + dt M_inv Fn
from math import sin, pi,exp
import numpy as np
import matplotlib.pyplot as plt

from utils import *

n_nodes = 11
n_elements = n_nodes - 1
h = 1/(n_nodes-1)

f = lambda x,t : (pi**2 - 1)*exp(-t)*sin(pi*x)

# connectivity given [element #][local node #] -> global node #
# there are (n_nodes - 1) elements, 2 nodes per element
connectivty = [(x,x+1) for x in range(0,n_elements)]

print(connectivty)

# parent functions & their derivatives on the interval [-1,1]
def phi_1(chi):
    return (1 - chi)/2

def phi_2(chi):
    return (chi + 1)/2

def phi_1_prime(chi):
    return -1/2

def phi_2_prime(chi):
    return 1/2

def x_to_chi(x,xi):
    '''mapping from global element location to parent element'''
    chi = (x - xi)*2/h -1
    return chi


dx_dchi = h/2 # jacobian for change of basis
dchi_dx = 2/h 

# global stiffness and mass matrices
K = np.zeros(shape = (n_nodes,n_nodes))
M = np.zeros(shape = (n_nodes,n_nodes))

# construct K & M by looping through the elements
k_local = np.zeros(shape = (2,2))
m_local = np.zeros(shape = (2,2))
for element_i in range(0,n_elements):

    #  local matrices
    for l in range(0,2):

        if l == 0:
            phi_l = phi_1
            phi_l_prime = phi_1_prime
        else:
            phi_l = phi_2
            phi_l_prime = phi_2_prime

        for m in range(0,2):
            
            if m == 0:
                phi_m = phi_1
                phi_m_prime = phi_1_prime
            else:
                phi_m = phi_2
                phi_m_prime = phi_2_prime

            # perform change of basis to calculate integral mass and stiffnes in parent space
            xi = element_i * h # x location of the ith node
            integrand_k = lambda chi: dchi_dx * phi_l_prime(chi) \
                                  * dchi_dx * phi_m_prime(chi) \
                                  * dx_dchi # jacobian 
            
            integrand_m = lambda chi: phi_l(chi) \
                                  * phi_m(chi) \
                                  * dx_dchi *2   # jacobian 
            
            k_local[l][m] = gauss_quadrature(integrand_k)
            m_local[l][m] = gauss_quadrature(integrand_m)

            
    # gloabl assembly
    for l in range(0,2):
        for m in range(0,2):
            global_node_1 = connectivty[element_i][l]
            global_node_2 = connectivty[element_i][m]
            K[global_node_1][global_node_2] += k_local[l][m]
            M[global_node_1][global_node_2] += m_local[l][m]

K[0,0]*=2
K[-1,-1]*=2
M[0,0]*=2
M[-1,-1]*=2

M_inv = np.linalg.inv(M)


# intial condition
u = np.sin(np.pi * np.linspace(0,1,n_nodes))
print("intial_u", u)
u[0] = 0
u[-1] = 0
#plt.plot(np.linspace(0,1,n_nodes),u,label = 'initial')

total_time = 1
n_steps = 552
delta_t = total_time/n_steps
t_elapsed = 0

F = np.zeros(n_nodes)
for step in range(0,n_steps):
    F = np.zeros(n_nodes)
    for k in range(0,n_elements):
        # find local elements
        flocal = [0, 0]
        xi = k * h
        integrand_1 = lambda chi: f((chi+1)*h/2 + xi, t_elapsed)*phi_1(chi)*dx_dchi
        integrand_2 = lambda chi: f((chi+1)*h/2 + xi, t_elapsed)*phi_2(chi)*dx_dchi
        flocal[0] = gauss_quadrature(integrand_1)
        flocal[1] = gauss_quadrature(integrand_2)
        #print(f"[DBG]: K = {k}, xi = {xi}, flocal =",flocal)
        # map
        for l in range(0,2):
            
            global_node = connectivty[k][l]
            F[global_node] += flocal[l]
            #print(f"\t[DBG connectivity] K = {k} L = {l} global = {global_node}")

    ###One way to do it?
    A = - delta_t * M_inv @ K
    b = delta_t * M_inv @ F
    # print("cond:", np.linalg.cond(A))
    dbc = True
    if dbc:
        A[0,:] = 0
        A[-1,:] = 0
        # b[0] = 0
        # b[-1] = 0
    u = u + A @ u + b
    # print("[DBG] F = ",F)
    # b = M@u-delta_t*K@u+F
    # u = np.linalg.solve(M, b)
    #print(F)
    t_elapsed += delta_t
    if step % 200 == 1:
        plt.plot(np.linspace(0,1,n_nodes),u,'--o', label = f't = {t_elapsed:.3f}',)

plt.plot(np.linspace(0,1,n_nodes),u,'--o', label = f't = {t_elapsed:.3f}',)
x=np.linspace(0,1,1000)
plt.plot(x, np.exp(-t_elapsed)*np.sin(np.pi * x), label = 'analytical soln')
plt.legend(bbox_to_anchor=(0.4, 0.8), loc="upper right")
plt.show()


#print(K)