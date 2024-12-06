from bilinear_mesh import BilinearUniformMesh
import numpy as np
import matplotlib.pyplot as plt

def solve_heat_equation(mesh: BilinearUniformMesh, u0, u_l, u_r, f, dt, tf, method):

    time_steps = np.arange(0,tf+dt,dt)

    U = np.zeros([len(time_steps),mesh.n_nodes])
    K = mesh.get_stiffness_matrix()
    M = mesh.get_mass_matrix()

    for i,t in enumerate(time_steps):

        # Set initial condition.
        if i == 0:
            U[0,:] = u0
            break
        
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

    return time_steps, U

def main():
    N = 11
    mesh = BilinearUniformMesh(N)
    u0 = np.sin( np.pi * mesh.get_node_locations())
    u_left = 0
    u_right = 0
    f = lambda x,t: (np.pi**2 - 1) * np.exp(-t) * np.sin(np.pi * x)
    dt = 1/551
    tf = 1

    t, U = solve_heat_equation(mesh, u0, u_left, u_right, f, dt, tf, 'forward_euler')

    plt.plot(mesh.get_node_locations(), U[-1,:], '--s',label = 'numerical')
    x=np.linspace(0,1,1000)
    plt.plot(x, np.exp(-t[-1])*np.sin(np.pi * x), label = 'analytical soln')    
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()