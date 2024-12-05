import numpy as np

def f(x, t):
    """
    Function to represent the original problem function.

    Args:
    x (float): Spatial variable.
    t (float): Time variable.

    Returns:
    float: Value of the function at (x, t).
    """
    return ((np.pi**2 - 1)*np.exp(-t)*np.sin(np.pi*x))

def basis_functions(h):
    """
    Generate basis functions and their derivatives for finite element method.

    Args:
    h (float): Mesh step size.

    Returns:
    tuple: List of basis functions (phi), dx/dz, dz/dx, derivatives of phi (dphi).
    """
    phi = [0,0]
    phi[0] = lambda z: (1-z)/2
    phi[1] = lambda z: (1+z)/2
    dxdz = h/2
    dzdx = 2/h
    dphi = [-1/2, 1/2]
    return phi, dxdz, dzdx, dphi

def get_matricies(N, nnElem, nt, h, iee):
    """
    Generate global stiffness, mass, and force matrices.

    Args:
    N (int): Number of nodes.
    nnElem (int): Number of nodes per element.
    nt (int): Number of timesteps.
    h (float): Mesh step size.
    iee (array): Element to node mapping.

    Returns:
    tuple: Stiffness matrix (K), mass matrix (M), force matrix (F).
    """
    K = np.zeros((N, N))
    M = np.zeros((N, N))
    F = np.zeros((N, nt + 1))

    klocal = np.zeros((nnElem, nnElem))
    mlocal = np.zeros((nnElem, nnElem))

    phi, dxdz, dzdx, dphi  = basis_functions(h)
    vals = [-1.0 / (3**0.5), 1.0 / (3**0.5)] # quad values
    weights = [1, 1] # quad weights

    ts = np.linspace(0, 1, nt+1)   

    for k in range(N-1):
        F[k, :] = (-1/8) * (f((vals[1]), ts) * phi[0](vals[1]) + f((vals[0]), ts) * phi[0](vals[0]))

        for l in range(nnElem): # 2 nodes per element, nnElem = number of nodes per element = 2
            global_node1 = int(iee[k][l])
            for m in range(nnElem):
                klocal[l][m] = (dphi[l]*dzdx*dphi[m]*dzdx*dxdz) * 2 # Quadrature but not dependant of time
                mlocal[l][m] = (weights[0]*phi[l](vals[0])*phi[m](vals[0]) + weights[1]*phi[l](vals[1])*phi[m](vals[1])) * dxdz * 2 # quadrature
                global_node2 = int(iee[k][m])
                K[global_node1][global_node2] += klocal[l][m]
                M[global_node1][global_node2] += mlocal[l][m]
    return K, M, F

if __name__ == '__main__':
    N = 11
    nnElem = 2
    nt = 1
    h = 1/(N-1)
    iee = np.vstack((np.arange(N - 1), np.arange(1, N))).T

    K,M,F = get_matricies(N,nnElem,nt,h,iee)

    print(F)