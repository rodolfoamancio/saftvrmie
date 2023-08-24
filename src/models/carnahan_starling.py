import numpy as np

class CarnahanStarling():
    def __init__(self, diameter: float):
        self.__diameter = diameter

    @property
    def diameter(self):
        return self.__diameter
    
    def packing_fraction(self, density: float) -> float:
        eta = density*np.pi*(self.diameter**3)/6
        return eta
    
    def helmholtz_energy(self, density: float) -> float:
        eta = self.packing_fraction(density)
        a_hs = (4*eta-3*(eta**2))/((1-eta)**2)
        return a_hs
    
    def compressibility_factor(self, density: float) -> float:
        eta = self.packing_fraction(density)
        k_hs = (
            ((1-eta)**4)
            *(1+4*eta+4*(eta*2)-4*(eta**3)+(eta**4))
        )
        return k_hs