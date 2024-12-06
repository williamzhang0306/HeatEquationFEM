from bilinear_mesh import BilinearUniformMesh
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Literal, Union

def solve_heat_equation(mesh: BilinearUniformMesh, 
                        u0: np.ndarray, 
                        u_l: Union [float, None], 
                        u_r: Union [float, None], 
                        f: Callable, 
                        dt:float, 
                        tf:float, 
                        method: Literal["forward_euler", "backward_euler"]):
    '''
    Solves the 1D heat equation with dirchlet boundary conditions using 
    the Galerkin finite element method.

    Arguments:
    - mesh (BilinearUniformMesh): Mesh object representing the spatial discretization.
    - u0 (np.ndarray): The initial conditions, a.k.a u(x,t=0).
    - u_l (float or None): The left essential boundary condition. If None, no B.C is applied.
    - u_r (float or None): Similarly the right essential b.c.
    - f (Callable): f(x,t) the forcing function. Must be callable as f(x,t)
    - dt (float): The time step size.
    - tf (float): Final time the solution is advanced to.

    Returns:
    - time_steps (np.ndarray): A vector of time steps solutions are evaluated at.
    - U (np.ndarray): A matrix of size (# timestep x # nodes), 
        U[i,j] = the solution at time step i and node j.
    '''

    time_steps = np.arange(0,tf+dt,dt)

    U = np.zeros([len(time_steps),mesh.n_nodes])
    K = mesh.get_stiffness_matrix()
    M = mesh.get_mass_matrix()

    for i,t in enumerate(time_steps):
        print(U[i,:])

        # Set initial condition.
        if i == 0:
            U[0,:] = u0
        
        # Don't calculate u(t+dt) if t=tf already.
        if t >= tf or i == len(time_steps)-1:
            break
        
        # estiamte u(t + dt)
        # rearrange Galerkin heat equation to A*u(t+dt) = b system and solve for u with numpy.
        if method == 'forward_euler':
            F = mesh.get_force_matrix(f,t)
            A = (1/dt) * M
            b = (1/dt)*M@U[i,:] - K@U[i,:] + F

            # essential (dirchlet) boundary conditions
            if u_l is not None:
                A[0,:] = 0
                A[0,0] = 1
                b[0] = u_l

            if u_r is not None:
                A[-1,:] = 0
                A[-1,-1] = 1
                b[-1] = u_r

            # Solve and Update
            U[i+1,:] = np.linalg.solve(A,b)

        elif method == 'backward_euler':
            F = mesh.get_force_matrix(f,t+dt) # implicit so use force at t+dt
            A = (1/dt)*M+K
            b = (1/dt)*M@U[i,:] + F
            
            # essential (dirchlet) boundary conditions
            if u_l is not None:
                A[0,:] = 0
                A[0,0] = 1
                b[0] = u_l

            if u_r is not None:
                A[-1,:] = 0
                A[-1,-1] = 1
                b[-1] = u_r

            # Solve and Update
            U[i+1,:] = np.linalg.solve(A,b)

        else:
            raise ValueError('invalid method')

    return time_steps, U