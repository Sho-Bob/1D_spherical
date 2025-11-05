import numpy as np
import matplotlib.pyplot as plt
class Tait:
    def __init__(self):
        self.B = 3.07*10**8 ## Pa
        self.p0 = 1.0e5
        self.rho0 = 1000.0 ## kg/m^3
        self.gamma = 7.5 ## parameter in Tait EOS
    
    def Get_P_from_rho(self,rho):
        """Compute pressure from density"""
        p = (self.p0 + self.B) * (rho/self.rho0)**self.gamma - self.B
        return p
    
    def Get_rho_from_P(self,p):
        """Compute density from pressure"""
        rho = self.rho0 * ((p + self.B) / (self.p0 + self.B))**(1.0/self.gamma)
        return rho
    
    def Get_sound_speed(self,rho):
        """Compute sound speed from density"""
        dpdrho = self.gamma * (self.p0 + self.B)*(rho/self.rho0)**(self.gamma-1.0)/self.rho0
        ss = np.sqrt(dpdrho)
        return ss


if __name__ == "__main__":
    tait = Tait()
    rho = np.linspace(0.01, 1000.0, 100)
    ss = np.zeros_like(rho)
    for i in range(len(rho)):
        ss[i] = tait.Get_sound_speed(rho[i])
        if(ss[i] <= 0.0):
            print(rho[i])
            raise ValueError('Sound speed is negative')
    plt.plot(rho, ss, 'k-', linewidth=2)
    plt.xlabel('Density (kg/m^3)')
    plt.ylabel('Sound speed (m/s)')
    plt.title('Sound speed vs Density')
    plt.grid(True)
    # plt.legend()
    plt.tight_layout()
    plt.show()