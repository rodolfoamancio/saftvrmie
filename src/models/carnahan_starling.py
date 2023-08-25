import numpy as np
from typing import Union

class CarnahanStarling():
    def __init__(self, diameter: Union[float, np.ndarray]):
        self.__diameter = diameter
        self.__phi = np.array([
            [7.5365557, -359.44, 1550.9, -1911.28, 9236.9, 10],
            [-37.60463, 1825.6, -5070.1, 9.063632, 21390.175, -129430, 10],
            [71.745953, -3168, 6534.6, -17.94482, -51320.7, 357230, 0.57],
            [-46.83552, 1884.2, -3288.7, 11.34027, 37064.54, -315530, -6.7],
            [-2.467982, -0.82376, -2.7171, 20.52142, 1103.742, 1390.2, -8],
            [-0.50272, -3.1935, 2.0883, -56.6377, -3264.61, 4518.2, 0],
            [8.0956883, 3.7090, 0, 40.53683, 2556.181, 4241.6, 0]
        ], dtype=float)

    @property
    def diameter(self) -> Union[float, np.ndarray]:
        return self.__diameter
    
    @property
    def phi(self) -> Union[float, np.ndarray]:
        return self.__phi
    
    def packing_fraction(self, density: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        eta = density*np.pi*(self.diameter**3)/6
        return eta
    
    def helmholtz_energy(self, density: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        eta = self.packing_fraction(density)
        a_hs = (4*eta-3*(eta**2))/((1-eta)**2)
        return a_hs
    
    def compressibility_factor(self, density: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        eta = self.packing_fraction(density)
        k_hs = (
            ((1-eta)**4)
            *(1+4*eta+4*(eta*2)-4*(eta**3)+(eta**4))
        )
        return k_hs
    
    def alpha(self, lambda_a: Union[float, np.ndarray], lambda_r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        C = (
            (lambda_r/(lambda_r - lambda_a))
            *((lambda_r/lambda_a)**(lambda_a/(lambda_r-lambda_a)))
        )
        alpha = C*((1/(lambda_a-3))-(1/(lambda_r-3)))
        return alpha

    def f_i(self, alpha: Union[float, np.ndarray], i: int) -> Union[float, np.ndarray]:
        phi_i = self.phi[i, :]
        f_i = (
            (phi_i[0] + phi_i[1]*(alpha**1) + phi_i[2]*(alpha**2) + phi_i[3]*(alpha**3))
            /(1 + phi_i[4]*alpha + phi_i[5]*(alpha**2) + phi_i[6]*(alpha**3))
        )
        return f_i