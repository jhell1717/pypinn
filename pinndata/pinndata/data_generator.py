import numpy as np

def burgers_1d_fd(u0, x, t_final, nu, dt):
    """
    Simulate viscous 1D Burgers equation
      u_t + u u_x = nu u_xx
    using 2nd-order central differences in space and RK4 in time.
    Periodic boundary conditions assumed.
    Returns: x (grid), t_grid (times), U (array shape (nt, nx))
    """
    nx = x.size
    dx = x[1] - x[0]
    nt = int(np.ceil(t_final / dt)) + 1
    U = np.zeros((nt, nx))
    U[0, :] = u0.copy()
    t_grid = np.linspace(0., t_final, nt)

    def spatial_rhs(u):
        # periodic indexing
        ux = (np.roll(u, -1) - np.roll(u, 1)) / (2*dx)         # u_x (central)
        uxx = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / (dx*dx) # u_xx
        return -u * ux + nu * uxx

    u = u0.copy()
    for n in range(1, nt):
        # RK4 steps
        k1 = spatial_rhs(u)
        k2 = spatial_rhs(u + 0.5*dt*k1)
        k3 = spatial_rhs(u + 0.5*dt*k2)
        k4 = spatial_rhs(u + dt*k3)
        u = u + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
        U[n, :] = u

    return x, t_grid, U
