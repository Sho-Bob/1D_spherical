import numpy as np
import matplotlib.pyplot as plt
from PengRobinson import PengRobinson as PR 

class SRK:
    def __init__(self, Tc, Pc, omega, M, R=8.314):
        """
        Init of SRK EOS class
        Tc: Critical temperature [K]
        Pc: Critical pressure [Pa]
        omega: Acentric factor 
        M: Molar mass [g/mol]
        R: Gas constant [J/(mol*K)], default is 8.314
        """
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.R = R
        self.Mw = M * 1e-3 # Convert to kg/mol
        self.Z = 0.0
        self.V = 0.0

        # Calculate SRK parameters a and b
        self.a = 0.42748 * (R * Tc)**2 / Pc
        self.b = 0.08664 * R * Tc / Pc
        self.kappa = 0.48 + 1.574 * omega - 0.176 * omega**2
    
    def Get_P_from_rho_and_T(self, rho, T):
        """
        Calculate pressure from density and temperature using SRK EOS.
        """
        alpha_T = self.alpha_from_T(T)
        molar_volume = self.Mw/ rho
        P = (self.R * T) / (molar_volume - self.b) - (self.a * alpha_T) / (molar_volume*(molar_volume + self.b))
        return P
    
    def Get_rho_from_P_and_T(self, P, T):
        """
        Calculate density from pressure and temperature using SRK EOS.   
        """
        self.calc_Z(P, T)
        rho = self.Mw / self.V
        return rho
    
    def alpha_from_T(self, T):
        Tr = T / self.Tc
        return (1.0 + self.kappa * (1.0 - np.sqrt(Tr)))**2
    
    def calc_Z(self, P, T):
        """
        Calculate compressibility factor Z by solving the SRK cubic equation.
        
        The cubic equation in Z for SRK is:
        Z^3 - Z^2 + (A - B - B^2)*Z - A*B = 0
        
        Where:
            A = a*alpha*P/(R*T)^2
            B = b*P/(R*T)
        """
        alpha_T = self.alpha_from_T(T)
        A = self.a * alpha_T * P / (self.R * T)**2
        B = self.b * P / (self.R * T)
        
        # Coefficients of cubic equation: Z^3 + c2*Z^2 + c1*Z + c0 = 0
        c2 = -1.0
        c1 = A - B - B**2
        c0 = -A * B
        
        # Solve cubic equation
        coeffs = [1.0, c2, c1, c0]
        roots = np.roots(coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        
        if len(real_roots) == 0:
            raise ValueError("No real roots found for compressibility factor")
        
        # Calculate fugacity coefficient for each root to select the correct phase
        # ln(phi) = Z - 1 - ln(Z - B) - (A/B) * ln(1 + B/Z)
        lnphi = np.zeros(real_roots.size)
        imin = 0
        for i in range(real_roots.size):
            Z_i = real_roots[i]
            if Z_i > B:  # Valid root (Z must be greater than B)
                lnphi[i] = Z_i - 1.0 - np.log(Z_i - B) - (A / B) * np.log(1.0 + B / Z_i)
                if i > 0 and lnphi[i] < lnphi[imin]:
                    imin = i
            else:
                lnphi[i] = np.inf  # Invalid root
        
        self.Z = real_roots[imin].real
        self.V = self.R * T * self.Z / P
    

# Example usage and test cases
if __name__ == "__main__":
    # Test with CO2
    Tc = 304.1 # K
    Pc = 7.3825e6
    omega = 0.225
    M = 44.01
    R = 8.314
    P = 8.0e6
    
    # Calculate density using SRK EOS
    T = np.linspace(200, 400, 100)
    rho_srk = np.zeros_like(T)
    rho_pr = np.zeros_like(T)
    srk = SRK(Tc, Pc, omega, M, R)
    pr = PR(Tc, Pc, omega, M, R)
    for i in range(len(T)):
        rho_srk[i] = srk.Get_rho_from_P_and_T(P, T[i])
        rho_pr[i] = pr.Get_rho_from_P_and_T(P, T[i])
    # Read NIST data
    nist_data_path = 'CO2_data_8MPa/NIST_data.xml'
    T_nist = []
    rho_nist = []
    
    with open(nist_data_path, 'r') as f:
        lines = f.readlines()
        # Skip the header line
        for line in lines[1:]:
            data = line.strip().split('\t')
            if len(data) >= 3:
                try:
                    temperature = float(data[0])  # Temperature in K
                    density = float(data[2])       # Density in kg/m³
                    T_nist.append(temperature)
                    rho_nist.append(density)
                except ValueError:
                    continue

    T_nist = np.array(T_nist)
    rho_nist = np.array(rho_nist)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(T_nist, rho_nist, 'r--', linewidth=2, label='NIST Data (8 MPa)')
    plt.plot(T, rho_srk, 'k-', linewidth=2, label='SRK EOS')
    plt.plot(T, rho_pr, 'k--', linewidth=2, label='Peng-Robinson EOS')
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Density (kg/m³)', fontsize=12)
    plt.xlim(220, 400)
    plt.title('CO₂ Density over Temperature at 8 MPa (SRK EOS)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()