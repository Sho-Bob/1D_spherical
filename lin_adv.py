import numpy as np
import matplotlib.pyplot as plt

"""This is a simple 1D linear advection FVM solver in spherical coordinate system by Sho Wada
   Equation:
   dphi/dt + 1/r^2 * d/dr(r^2 * ur * phi) = 0
"""

def main():
    """
    r is the grid locations, fr is the flux locations
    | o | o | o ... o |
    o : grid location
    | : flux location

    grid location starts from r = delta r 
    flux location starts from r = 0.0

    At each flux location, we have two values of phi and ur. 
    phi_fL and phi_fR, ur_fL and ur_fR. The following fig is one cell with two flux locations (L and R included)

       L R               L R
    ____|_______o_________|____
            cell center
    """

    # Set up grids and time
    nt = 300       ## number of time steps
    nfr = 101      ## number of flux locations
    nr = nfr - 1   ## number of grid locations
    CFL = 0.5      ## CFL number to determine time step size
    fr = np.linspace(0.0, 1.0, nfr) ## flux locations
    r = np.zeros(nr)                ## grid locations
    dV = np.zeros(nr)               ## Volume of the one cell
    time = np.zeros(nt)             ## time array
    
    # Compute grid locations and volumes
    r[0] = 0.5 * (fr[0] + fr[1])
    for i in range(1, nr):
        r[i] = 0.5 * (fr[i] + fr[i+1])
        dV[i] = 1.0/3.0 * (fr[i+1]**3 - fr[i]**3)
    dV[0] = 1.0/3.0 * (fr[1]**3 - fr[0]**3)

    # Define variables (linear advection equation)
    ur = np.ones(nr)
    phi = np.zeros(nr)
    phi_new = phi.copy()

    # Initialization of phi for now, Gaussian pulse
    phi = np.exp(-(r - 0.5)**2 / 0.1**2)
    phi_initial = phi.copy()

    # Time loop starts here
    for itr in range(1,nt):
        
        # dt = CFL * np.min(dV) / np.max(np.abs(ur))
        dt = 0.001
        time[itr] = time[itr-1] + dt

        # Explicit Euler method    
        # Compute the values of phi and ur at the flux locations
        ur_fL, phi_fL, ur_fR, phi_fR = compute_flux_values(ur, phi)
        
        # Compute the flux
        flux = compute_flux(ur_fL, phi_fL, ur_fR, phi_fR, fr)
        
        # Compute the rhs
        rhs = compute_rhs(flux, dV, dt)
        
        # Update phi
        phi_new = phi + rhs

        phi = phi_new.copy()
    

    # Compute the analytical solution
    phi_analytical = phi_initial.copy()
    for i in range(nr):
        phi_analytical[i] = ((r[i]-time[itr-1])/r[i])**2 * np.exp(- (r[i]-time[itr-1]-0.5)**2 / 0.1**2)
    
    # Plot the results
    plt.plot(r, phi,'k-',label=f'time = {time[itr]:.2f}')
    plt.plot(r, phi_initial,'k--', label='initial')
    plt.plot(r, phi_analytical,'r-', label='analytical')
    plt.xlabel('r')
    plt.ylabel('phi')
    plt.grid(True)
    plt.xlim(0, 1)
    plt.title('Linear advection equation')
    plt.legend(loc='upper right')
    plt.show()

def compute_flux_values(ur, phi):
   """ This function computes the values of phi and ur at the flux locations
       For now, its just first order reconstruction.
       ur_fL[0] and ur_fR[-1] are out of the computational domain, so we have to use the boundary conditons.
       So as phi_fL[0] and phi_fR[-1]
   """
   ndr = ur.shape[0]
   nfr = ndr + 1
   ur_fL = np.zeros(nfr)
   phi_fL = np.zeros(nfr)
   ur_fR = np.zeros(nfr)
   phi_fR = np.zeros(nfr)

   for i in range(1,nfr-1):
       ur_fL[i] = ur[i-1]
       phi_fL[i] = phi[i-1]
       ur_fR[i] = ur[i]
       phi_fR[i] = phi[i]

   ur_fL[0] = ur[0] ## Neuman BC
   phi_fL[0] = phi[0] ## Neuman BC
   ur_fR[0] = ur[0] ## 1st order
   ur_fR[nfr-1] = ur[ndr-1] ## Neuman BC
   phi_fR[nfr-1] = phi[ndr-1] ## Neuman BC

   return ur_fL, phi_fL, ur_fR, phi_fR

def compute_flux(ur_fL, phi_fL, ur_fR, phi_fR, fr):
   """ This function computes the flux at the flux locations"""
   nfr = len(fr)
   flux_upwind = np.zeros(nfr)
   flux_central = np.zeros(nfr)
   
   flag_upwind = False
   
   ### Assuming the advection speed is constant for now
   for i in range(nfr):
       velo = np.abs(0.5*(ur_fL[i] + ur_fR[i]))

       # This is upwind scheme
       flux_upwind[i] = fr[i]**2 * (0.5 * (ur_fL[i]*phi_fL[i] + ur_fR[i]*phi_fR[i]) - 0.5 * velo * (phi_fR[i] - phi_fL[i]))
       # This is central scheme
       flux_central[i] = fr[i]**2 * 0.5 * (ur_fL[i]*phi_fL[i] + ur_fR[i]*phi_fR[i])
    
   if flag_upwind:
       return flux_upwind
   else:
       return flux_central
   

def compute_rhs(flux, dV, Delta_t):
   """ This function computes the rhs of the linear advection equation"""
   nfr = len(flux)
   number_of_cells = len(dV)
   if nfr != number_of_cells+1:
    raise ValueError("The number of flux locations and the number of grid locations are not consistent")
   
   rhs = np.zeros(number_of_cells)
   for i in range(number_of_cells):
       rhs[i] = (flux[i+1] - flux[i]) / dV[i]
   return -rhs*Delta_t

if __name__ == "__main__":
    main()