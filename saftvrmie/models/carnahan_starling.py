import numpy as np
from typing import Union

class CarnahanStarling():
    def __init__(self, diameter: Union[float, np.ndarray]):
        self.__diameter = diameter
        self.__phi = np.array([
            [7.5365557, -359.44, 1550.9, -1.19932, -1911.28, 9236.9],
            [-37.60463, 1825.6, -5070.1, 9.063632, 21390.175, -129430],
            [71.745953, -3168, 6534.6, -17.94482, -51320.7, 357230],
            [-46.83552, 1884.2, -3288.7, 11.34027, 37064.54, -315530],
            [-2.467982, -0.82376, -2.7171, 20.52142, 1103.742, 1390.2],
            [-0.50272, -3.1935, 2.0883, -56.6377, -3264.61, -4518.2],
            [8.0956883, 3.7090, 0, 40.53683, 2556.181, 4241.6]
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
    
    def alpha(self, lambda_a: int, lambda_r: int) -> float:
        C = (
            (lambda_r/(lambda_r - lambda_a))
            *((lambda_r/lambda_a)**(lambda_a/(lambda_r-lambda_a)))
        )
        alpha = C*((1/(lambda_a-3))-(1/(lambda_r-3)))
        return alpha

    def __alpha_powers(self, alpha: float) -> np.ndarray:
        alpha = np.array([alpha])
        powers = np.arange(4)
        alpha_col = alpha[:, np.newaxis]
        result = alpha_col**powers
        return result.T
    
    def __f(self, alpha: float) -> np.ndarray:
        alpha_powers = self.__alpha_powers(alpha)
        # (6 x 4) x (4, ) = (6, )
        numerator = np.matmul(self.phi[:4, :].T*alpha_powers)
        # (6 x 3) x (3, ) = (6, )
        demonimator = 1 + np.matmul(self.phi[4:, :].T*alpha_powers[1:])

        # final result, shape: (6, )
        f = (numerator/demonimator)
        return f
    
    def correction_factor(self, alpha: float, packing_fraction: Union[float, np.ndarray], x0: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(packing_fraction, np.ndarray):
            packing_fraction = packing_fraction[:, np.newaxis]

        f = self.__f(alpha)

        correction_factor = (
            f[0]*packing_fraction*(x0**3)
            +f[3]*((packing_fraction*(x0**3))**5)
            +f[4]*((packing_fraction*(x0**3))**8)
        )
        return correction_factor