import numpy as np
import matplotlib.pyplot as plt
from PengRobinson import PengRobinson as PR 

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
    nt = 1500      ## number of time steps
    nfr = 101      ## number of flux locations
    nr = nfr - 1   ## number of grid locations
    CFL = 0.3      ## CFL number to determine time step size
    terminal_output_interval = 100  ## number of time steps to output the results to the terminal
    time_step = 0
    fr = np.linspace(0.0, 1.0, nfr) ## flux locations
    r = np.zeros(nr)                ## grid locations
    flag_analytic = False           ## Flag to compute the analytical solution
    flag_upwind = True              ## Flag to use upwind scheme
    dV = np.zeros(nr)               ## Volume of the one cell
    dr = np.zeros(nr)               ## Distance between grid locations
    time = np.zeros(nt)             ## time array
    Time_step_method = 'RK3'        ## Time step method: Euler or RK3

    ### CO2 critical properties
    Tc = 304.1 # K
    Pc = 7.3825e6
    omega = 0.225
    M = 44.01
    R = 8.314
    pr = PR(Tc, Pc, omega, M, R)

    # Compute grid locations and volumes
    r[0] = 0.5 * (fr[0] + fr[1])
    for i in range(1, nr):
        r[i] = 0.5 * (fr[i] + fr[i+1])
        dV[i] = 1.0/3.0 * (fr[i+1]**3 - fr[i]**3)
        dr[i] = fr[i+1] - fr[i]
    dV[0] = 1.0/3.0 * (fr[1]**3 - fr[0]**3)
    dr[0] = fr[1] - fr[0]

    # Define variables (Euler equation)
    rho = np.ones(nr)
    rhou = np.zeros(nr)
    ur = np.zeros(nr)
    p = np.zeros(nr)
    rho_old = rho.copy()
    rhou_old = rhou.copy()
    rho_new = rho.copy()
    rhou_new = rhou.copy()
    T = 293.0 ## Isothermal
    R = 8.314 ## Gas constant
    gamma = 1.4 ## Adiabatic index
    mu = 1.0e-5 ## Dynamic viscosity: mu is assumed to be constant for now
    p_ini = 14.7 * 1e6 ## 147 bar

    # Initialization of phi for now, Gaussian pulse
    # rho = np.exp(-(r - 0.5)**2 / 0.1**2)
    rho_ini = pr.Get_rho_from_P_and_T(p_ini,T)
    rho = rho_ini * np.ones(nr)
    p = p_ini * np.ones(nr)
    rhou = rho * ur

    rho_initial = rho.copy()
    p_initial = p.copy()

    # Time loop starts here
    for itr in range(1,nt):
        time_step += 1
        ## dt calc. with CFL 
        sound_speed = Comp_sound_speed(rho,T)
        max_wave_speed = np.max(np.abs(ur) + sound_speed)
        dt = compute_dt_from_CFL(CFL, max_wave_speed, dr, mu)
        time[itr] = time[itr-1] + dt

        if Time_step_method == 'Euler':
            # Explicit Euler method    
            # Compute the values of phi and ur at the flux locations
            # Reconstruct only primitive variables (rho, ur, P)
            ur_fL, rho_fL, p_fL, ur_fR, rho_fR, p_fR = compute_flux_values(ur, rho, p)
            
            # Compute the flux
            flux = compute_flux(max_wave_speed, ur_fL, rho_fL, p_fL, ur_fR, rho_fR, p_fR, fr, flag_upwind)
            source_flux = compute_source_flux(r,fr,p_fL,p_fR,dV)

            # Compute the diffusion term in the FDM manner
            diffusion_term = compute_diffusion_term(ur,mu,fr,r,dr)
            # diffusion_term = np.zeros(nr)
            
            # Compute the rhs
            rhs = compute_rhs(flux, source_flux, diffusion_term, p_fL, p_fR, dV, fr, r, dt)
            
            # Update conservative variables
            rho_new = rho + rhs[0,:]
            rhou_new = rhou + rhs[1,:]

            # Update primitive variables
            p_new = rho_new * R * T ## PR EOS later
            ur_new = rhou_new / rho_new

            rho = rho_new.copy()
            rhou = rhou_new.copy()
            ur = ur_new.copy()
            p = p_new.copy()
        
        elif Time_step_method == 'RK3':
            rho_old = rho.copy()
            rhou_old = rhou.copy()
            for rk_step in range(3):
                sound_speed = np.sqrt(gamma * R * T)
                max_wave_speed = np.max(np.abs(ur) + sound_speed)
                ur_fL, rho_fL, p_fL, ur_fR, rho_fR, p_fR = compute_flux_values(ur, rho, p)
                flux = compute_flux(max_wave_speed, ur_fL, rho_fL, p_fL, ur_fR, rho_fR, p_fR, fr, flag_upwind)
                source_flux = compute_source_flux(r,fr,p_fL,p_fR,dV)
                diffusion_term = compute_diffusion_term(ur,mu,fr,r,dr)
                rhs = compute_rhs(flux, source_flux, diffusion_term, p_fL, p_fR, dV, fr, r, dt)
                if rk_step == 0:
                    rho_new = rho_old + rhs[0,:]
                    rhou_new = rhou_old + rhs[1,:]
                elif rk_step == 1:
                    rho_new = 0.75 * rho_old + 0.25 * (rho + rhs[0,:])
                    rhou_new = 0.75 * rhou_old + 0.25 * (rhou + rhs[1,:])
                elif rk_step == 2:
                    rho_new = 1.0/3.0 * rho_old + 2.0/3.0 * (rho + rhs[0,:])
                    rhou_new = 1.0/3.0 * rhou_old + 2.0/3.0 * (rhou + rhs[1,:])
                
                # Update primitive variables
                p_new = rho_new * R * T ## PR EOS later
                ur_new = rhou_new/rho_new

                # Update conservative variables
                rho = rho_new.copy()
                rhou = rhou_new.copy()
                ur = ur_new.copy()
                p = p_new.copy()

        
        # Output the results to the terminal
        if time_step % terminal_output_interval == 0:
            output_terminal(time_step, rho, ur, p, time[itr])
    
    # Compute the analytical solution
    if flag_analytic:
        rho_analytical = rho_initial.copy()
        for i in range(nr):
            rho_analytical[i] = ((r[i]-time[itr-1])/r[i])**2 * np.exp(- (r[i]-time[itr-1]-0.5)**2 / 0.1**2)

    # Plot the results
    # plt.plot(r, p,'k-',label=f'time = {time[itr]:.2f}')
    # plt.plot(r, p_initial,'k-.', label='initial')
    plt.plot(r, rho,'k-',label=f'time = {time[itr]:.2f}')
    plt.plot(r, rho_initial,'k-.', label='initial')
    if flag_analytic:
        plt.plot(r, rho_analytical,'r--', label='analytical')
    plt.xlabel('r')
    plt.ylabel('rho')
    # plt.ylabel('Pressure')
    plt.grid(True)
    plt.xlim(0, np.max(r))
    plt.title('Euler equations in spherical coordinate system')
    plt.legend(loc='upper right')
    # plt.savefig(f'./data/Euler_rho_time_{time[itr]:.2f}.png')
    plt.show()


def compute_flux_values(ur, rho, p):
   """ This function computes the values of phi and ur at the flux locations
       For now, its just first order reconstruction.
       ur_fL[0] and ur_fR[-1] are out of the computational domain, so we have to use the boundary conditons.
       So as rho_fL[0] and rho_fR[-1] and p_fL[0] and p_fR[-1]
   """
   ndr = ur.shape[0]
   nfr = ndr + 1
   ur_fL = np.zeros(nfr)
   rho_fL = np.zeros(nfr)
   p_fL = np.zeros(nfr)
   ur_fR = np.zeros(nfr)
   rho_fR = np.zeros(nfr)
   p_fR = np.zeros(nfr)
   
   # Can change the order of accuracy in space later by using MUSCL scheme etc.
   for i in range(1,nfr-1):
       ur_fL[i] = ur[i-1]
       rho_fL[i] = rho[i-1]
       p_fL[i] = p[i-1]
       ur_fR[i] = ur[i]
       rho_fR[i] = rho[i]
       p_fR[i] = p[i]

   # Neumann BC
#    ur_fL[0] = ur[0]

   ## Dirichlet BC
   ur_fL[0] = 0.0
   rho_fL[0] = rho[0] ## Neuman BC
   p_fL[0] = p[0] ## Neuman BC
   rho_fR[0] = rho[0] ## 1st order
   p_fR[0] = p[0] ## 1st order
   ur_fL[nfr-1] = ur[ndr-1] ## 1st order
   rho_fL[nfr-1] = rho[ndr-1] ## 1st order
   p_fL[nfr-1] = p[ndr-1] ## 1st order
   ur_fR[nfr-1] = ur[ndr-1] ## Neuman BC
   rho_fR[nfr-1] = rho[ndr-1] ## Neuman BC
   p_fR[nfr-1] = p[ndr-1] ## Neuman BC

   return ur_fL, rho_fL, p_fL, ur_fR, rho_fR, p_fR


def compute_dt_from_CFL(CFL, max_wave_speed, dr, mu):
    """ This function computes the time step size from the CFL number"""
    dt_conv = CFL * np.min(dr) / max_wave_speed
    dt_visc = CFL * np.min(dr)**2 / mu
    return min(dt_conv, dt_visc)

def compute_flux(max_wave_speed, ur_fL, rho_fL, p_fL, ur_fR, rho_fR, p_fR, fr, flag_upwind):
   """ This function computes the flux at the flux locations"""
   nfr = len(fr)
   flux_upwind = np.zeros((2,nfr))
   flux_central = np.zeros((2,nfr))   
   
   ## For upwind scheme, Lax-Friedrichs scheme is used
   for i in range(nfr):
       # This is upwind scheme
       flux_upwind[0,i] = fr[i]**2 * (0.5 * (ur_fL[i]*rho_fL[i] + ur_fR[i]*rho_fR[i]) - 0.5 * max_wave_speed * (rho_fR[i] - rho_fL[i]))
       flux_upwind[1,i] = fr[i]**2 * (0.5 * (ur_fL[i]**2*rho_fL[i] + p_fL[i] + ur_fR[i]**2*rho_fR[i] + p_fR[i]) - 0.5 * max_wave_speed * (ur_fR[i]*rho_fR[i] - ur_fL[i]*rho_fL[i]))
       # This is central scheme
       flux_central[0,i] = fr[i]**2 * 0.5 * (ur_fL[i]*rho_fL[i] + ur_fR[i]*rho_fR[i])
       flux_central[1,i] = fr[i]**2 * 0.5 * (ur_fL[i]**2*rho_fL[i] + p_fL[i] + ur_fR[i]**2*rho_fR[i] + p_fR[i])
    
   if flag_upwind:
       return flux_upwind
   else:
       return flux_central


def compute_source_flux(r,fr,p_fL,p_fR,dV):
    """ This function computes the source flux for momentum equation at the flux locations"""
    nfr = len(fr)
    source_flux = np.zeros(nfr)
    ## For the source flux, just use the central scheme
    for i in range(nfr):
        source_flux[i] = fr[i]**2 *0.5 * (p_fL[i] + p_fR[i]) 
    return source_flux

def compute_diffusion_term(ur,mu,fr,r, dr):
    """ This function computes the diffusion term in the FDM manner"""
    nr = ur.shape[0]
    nfr = fr.shape[0]
    if nr != nfr - 1:
        raise ValueError("The number of grid locations and the number of flux locations are not consistent")
    diffusion_term = np.zeros(nr)
    durdr = np.zeros(nfr)
    ur_at_flux = np.zeros(nfr)

    # Compute the values of ur and durdr at flux points
    # Flux location i is between grid points i-1 and i
    # Distance between grid points i-1 and i is 0.5*(dr[i] + dr[i-1]) = 0.5*(fr[i+1] - fr[i-1])
    for i in range(1,nfr-1):
        ur_at_flux[i] = 0.5 * (ur[i-1] + ur[i])
        durdr[i] = (ur[i] - ur[i-1]) / (r[i] - r[i-1])
    durdr[0] = ur[0]/r[0]
    durdr[nfr-1] = 0.0    # Neumann BC
    ur_at_flux[0] = ur[0] # Neumann BC
    ur_at_flux[nfr-1] = ur[nr-1] # Neumann BC

    # Compute the diffusion term
    # Use cell widths (dr) for the finite difference
    for i in range(nr):
        cell_width = dr[i]  # fr[i+1] - fr[i]
        diffusion_term[i] = 4.0/3.0 / r[i]**2 * ((mu * fr[i+1]**2*durdr[i+1] - mu*fr[i]**2*durdr[i])/cell_width - (mu*fr[i+1]*ur_at_flux[i+1] - mu*fr[i]*ur_at_flux[i])/cell_width)
    
    return diffusion_term   

def Comp_sound_speed(rho,T):
    """ This function computes the sound speed using PR EOS"""
    Tc = 304.1 # K
    Pc = 7.3825e6
    omega = 0.225
    M = 44.01
    R = 8.314
    P = 8.0e6
    pr = PR(Tc, Pc, omega, M, R)
    ndr = np.shape(rho)[0]
    sound_speed = np.zeros(ndr)
    for i in range(ndr):
        pr.Compute_cp_and_cv_from_rho_and_T(rho[i],T)
        sound_speed[i] = pr.Get_sound_speed()
    return sound_speed

def compute_rhs(flux, source_flux, diffusion_term, p_fL, p_fR, dV, fr, r,Delta_t):
   """ This function computes the rhs of the Euler equations"""
   nfr = flux.shape[1]
   number_of_cells = len(dV)
   if nfr != number_of_cells+1:
    raise ValueError("The number of flux locations and the number of grid locations are not consistent")
   
   rhs = np.zeros((2,number_of_cells))
   flag_force = True
   for i in range(number_of_cells):
       rhs[0,i] = (flux[0,i+1] - flux[0,i]) / dV[i]
       if flag_force:
        rhs[1,i] = (flux[1,i+1] - flux[1,i]) / dV[i] - (source_flux[i+1] - source_flux[i]) / dV[i] - (p_fR[i] - p_fL[i]) / (fr[i+1] - fr[i]) - diffusion_term[i] - 1e-2/r[i]**2 
       else:
        rhs[1,i] = (flux[1,i+1] - flux[1,i]) / dV[i] - (source_flux[i+1] - source_flux[i]) / dV[i] - (p_fR[i] - p_fL[i]) / (fr[i+1] - fr[i]) - diffusion_term[i]
   return -rhs*Delta_t

def output_terminal(time_step, rho, ur, p, time):
    """ This function outputs the results to the terminal"""
    print(f'--------------------------------')
    print(f'Time step: {time_step}')
    print(f'Time: {time}')
    print(f'Max rho: {np.max(rho)}')
    print(f'Min rho: {np.min(rho)}')
    print(f'Max ur: {np.max(ur)}')
    print(f'Min ur: {np.min(ur)}')
    print(f'Max p: {np.max(p)}')
    print(f'Min p: {np.min(p)}')
    print(f'--------------------------------')

if __name__ == "__main__":
    main()