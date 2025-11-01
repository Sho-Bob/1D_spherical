import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
class PengRobinson:
    def __init__(self, Tc, Pc, omega, M, R=8.314):
        """
        Init of PengRobinson EOS class
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
        self.dpdv = 0.0
        self.dpdt = 0.0
        self.cv = 0.0
        self.cp = 0.0
        self.h = 0.0
        self.s = 0.0
        self.rho = 0.0

        # Calculate PR parameters a and b
        self.a = 0.45724 * (R * Tc)**2 / Pc
        self.b = 0.07780 * R * Tc / Pc
        self.kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    
    def Get_P_from_rho_and_T(self, rho, T):
        """
        Calculate pressure from density and temperature using PR EOS.
        """
        alpha_T = self.alpha_from_T(T)
        molar_volume = self.Mw/ rho
        P = (self.R * T) / (molar_volume - self.b) - (self.a * alpha_T) / (molar_volume**2 + 2*self.b*molar_volume - self.b**2)
        return P
    
    def Get_rho_from_P_and_T(self, P, T):
        """
        Calculate density from pressure and temperature using PR EOS.   
        """
        self.calc_Z(P, T)
        self.rho = self.Mw / self.V
        return self.rho
    
    def alpha_from_T(self, T):
        Tr = T / self.Tc
        return (1.0 + self.kappa * (1.0 - np.sqrt(Tr)))**2
    
    def dalpha_dT_from_T(self, T):
        Tr = T / self.Tc
        return - (1.0 + self.kappa * (1.0-np.sqrt(Tr))) * self.kappa/np.sqrt(self.Tc*T)

    def d2alpha_dT2_from_T(self, T):
        Tr = T / self.Tc
        w = 1.0 + self.kappa * (1.0 - np.sqrt(Tr))
        # return (self.kappa / (2.0 * T)) * (self.kappa / self.Tc + w / np.sqrt(self.Tc * T)) 
        return 0.5 * T**(-1.5) * self.kappa/np.sqrt(self.Tc) * w + self.kappa**2/(2.0*self.Tc*T)
    
    def calc_Z(self, P, T):
        """
        Calculate compressibility factor Z by solving the PR cubic equation.
        
        The cubic equation in Z is:
        Z^3 + (B-1)*Z^2 + (A - 3*B^2 - 2*B)*Z + (B^3 + B^2 - A*B) = 0
    
        """
        alpha_T = self.alpha_from_T(T)
        A = self.a * alpha_T * P / (self.R * T)**2
        B = self.b * P / (self.R * T)
        
        # Coefficients of cubic equation: Z^3 + c2*Z^2 + c1*Z + c0 = 0
        c2 = B - 1.0
        c1 = A - 3.0*B**2 - 2.0*B
        c0 = B**3 + B**2 - A*B
        
        # Solve cubic equation
        coeffs = [1.0, c2, c1, c0]
        roots = np.roots(coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        
        if len(real_roots) == 0:
            raise ValueError("No real roots found for compressibility factor")
        
        lnphi = np.zeros(real_roots.size)
        imin = 0
        for i in range(real_roots.size):
            V = self.R * T * real_roots[i] / P
            lnphi[i] = real_roots[i] - 1.0 - np.log(real_roots[i] - B) - A / B * np.log(1.0 + B / real_roots[i])
            if i > 0 and lnphi[i] < lnphi[imin]:
                imin = i
        self.Z = real_roots[imin].real
        self.V = self.R * T * self.Z / P
    
    def Compute_ideal_gas_prop(self,T):
        # --- NASA coefficients for CO2 (TP-2002-211556) ---
        coeff_low  = np.array([3.85746029E+00, 4.41437026E-03, -2.21481404E-06,
                            5.23490188E-10, -4.72084164E-14, -4.87591660E+04, 2.27163806E+01])
        coeff_high = np.array([2.35677352E+00, 8.98459677E-03, -7.12356269E-06,
                            2.45919022E-09, -1.43699548E-13, -4.83719697E+04, 9.90105222E+00])
        T_mid = 1000.0  # K
        # --- Gas constant --- 
        R = self.R / self.Mw  # J/(kg*K)

        # Depending on T, choose low- or high-T coefficients
        a = np.where(np.atleast_1d(T) < T_mid, coeff_low[:, None], coeff_high[:, None]) if np.ndim(T) else \
            (coeff_low if T < T_mid else coeff_high)

        # Evaluate polynomial
        T = np.array(T, ndmin=1)
        Cp_R = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4
        H_RT = a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T
        S_R  = a[0]*np.log(T) + a[1]*T + a[2]*T**2/2 + a[3]*T**3/3 + a[4]*T**4/4 + a[6]

        self.cp = Cp_R * R
        self.cv = (Cp_R - 1) * R
        self.h  = H_RT * R * T
        self.s  = S_R * R
    
    def Compute_cp_and_cv_from_rho_and_T(self,rho,T):
        self.Compute_ideal_gas_prop(T)
        R = self.R/self.Mw
        self.rho = rho
        self.V = self.Mw / self.rho
        self.dpdv = -self.R*T/(self.V-self.b)**2 + self.a*self.alpha_from_T(T)*2.0*(self.V+self.b)/(self.V**2+2.0*self.V*self.b-self.b**2)**2
        dalphadT = self.dalpha_dT_from_T(T)
        d2alphadT2 = self.d2alpha_dT2_from_T(T)
        K1 = 1.0/(np.sqrt(8.0)*self.b) * np.log((self.V+(1.0-np.sqrt(2.0))*self.b)/(self.V+(1.0+np.sqrt(2.0))*self.b))
        
        self.dpdt = self.R/(self.V-self.b) - self.a*dalphadT/(self.V**2+2.0*self.V*self.b-self.b**2)
        dep_cp = - ( K1*T*d2alphadT2*self.a + T*self.dpdt**2/self.dpdv) # [J/(mol*K)]
        dep_cp = dep_cp /self.Mw # [J/(kg*K)]
        dpdrho = -(self.V**2/self.Mw) * self.dpdv
        self.cp = self.cp + dep_cp
        drhodT = (self.Mw/self.V**2) * (self.dpdt/self.dpdv)
        alpha_rho = -1.0/self.rho * drhodT
        kappa_T = 1.0/(self.rho * dpdrho)
        self.cv = self.cp - T/self.rho *alpha_rho**2/kappa_T
        self.sound_speed = np.sqrt(self.cp/self.cv * dpdrho)
    
    def Get_sound_speed(self):
        return self.sound_speed
    
# Example usage and test cases
if __name__ == "__main__":
    # Test with CO2
    Tc = 304.1 # K
    Pc = 7.3825e6
    omega = 0.225
    M = 44.01
    R = 8.314
    P = 8.0e6
    
    # Calculate density using Peng-Robinson EOS
    T_pr = np.linspace(200, 400, 100)
    rho_pr = np.zeros_like(T_pr)
    cp_pr = np.zeros_like(T_pr)
    sos_pr = np.zeros_like(T_pr)
    pr = PengRobinson(Tc, Pc, omega, M, R)
    for i in range(len(T_pr)):
        rho_pr[i] = pr.Get_rho_from_P_and_T(P, T_pr[i])
        pr.Compute_cp_and_cv_from_rho_and_T(rho_pr[i], T_pr[i])
        p_comp = pr.Get_P_from_rho_and_T(rho_pr[i], T_pr[i])
        if p_comp!=P:
            print(f'Error: p_comp = {p_comp} != P = {P}')
        cp_pr[i] = pr.cp
        sos_pr[i] = pr.sound_speed
    
    # Read NIST data
    nist_data_path = 'CO2_data_8MPa/NIST_data.xml'
    T_nist = []
    rho_nist = []
    cp_nist = []
    cv_nist = []
    sos_nist = []
    with open(nist_data_path, 'r') as f:
        lines = f.readlines()
        # Skip the header line
        for line in lines[1:]:
            data = line.strip().split('\t')
            if len(data) >= 3:
                try:
                    temperature = float(data[0])  # Temperature in K
                    density = float(data[2])       # Density in kg/m³
                    cp = float(data[8])
                    cv = float(data[7])
                    T_nist.append(temperature)
                    rho_nist.append(density)
                    cp_nist.append(cp)
                    cv_nist.append(cv)
                    sos_nist.append(float(data[9]))
                except ValueError:
                    continue
    
    # Convert lists to numpy arrays for plotting
    T_nist = np.array(T_nist)
    rho_nist = np.array(rho_nist)
    cp_nist = np.array(cp_nist)
    cv_nist = np.array(cv_nist)
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(T_nist, rho_nist, 'b-', linewidth=2, label='NIST Data (8 MPa)')
    plt.plot(T_pr, rho_pr, 'r--', linewidth=2, label='Peng-Robinson EOS')
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Density (kg/m³)', fontsize=12)
    plt.xlim(220, 400)
    plt.title('CO₂ Density vs Temperature at 8 MPa', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.plot(T_nist, cp_nist*1000, 'k--', linewidth=2, label='NIST Data (8 MPa)')
    # plt.plot(T_pr, cp_pr, 'k-', linewidth=2, label='Peng-Robinson EOS')
    # plt.xlabel('Temperature (K)', fontsize=12)
    # plt.ylabel('Cp (J/(kg*K))', fontsize=12)
    # plt.xlim(220, 400)
    # plt.title('CO₂ Cp vs Temperature at 8 MPa', fontsize=14)
    # plt.grid(True, alpha=0.3)
    # plt.legend(fontsize=11)
    # plt.tight_layout()
    # plt.show()

    plt.plot(T_nist, sos_nist, 'k--', linewidth=2, label='NIST Data (8 MPa)')
    plt.plot(T_pr, sos_pr, 'k-', linewidth=2, label='Peng-Robinson EOS')
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Sound Speed (m/s)', fontsize=12)
    plt.xlim(220, 400)
    plt.title('CO₂ Sound Speed vs Temperature at 8 MPa', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()