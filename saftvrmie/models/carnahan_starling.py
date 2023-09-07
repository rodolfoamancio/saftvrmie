import numpy as np
from typing import Union

from saftvrmie.constants.constants import ANGSTRON

class CarnahanStarling():
    """
    Class for Carnahan-Starling equation of state calculations.
    Based on Lafitte, Thomas, et al. "Accurate statistical associating fluid theory 
    for chain molecules formed from Mie segments." The Journal of chemical physics 139.15 (2013).
    """
    def __init__(self, diameter: Union[float, np.ndarray]):
        """
        Initialize the Carnahan-Starling equation of state parameters.

        Parameters:
        - diameter: float or np.ndarray - The diameter(s) of the particles.
        """
        self.__diameter = diameter

    @property
    def diameter(self) -> Union[float, np.ndarray]:
        """
        Get the diameter(s) of the particles.
        """
        return self.__diameter
        
    def packing_fraction(self, density: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the packing fraction.

        Parameters:
        - density: float or np.ndarray - The value(s) of the density.

        Returns:
        - eta: float or np.ndarray - The calculated packing fraction.

        Reference: Equation 11 details from Lafitte, 2013.
        """
        eta = density * np.pi * ((self.diameter) ** 3) / 6
        return eta
    
    def helmholtz_energy(self, density: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the Helmholtz energy.

        Parameters:
        - density: float or np.ndarray - The value(s) of the density.

        Returns:
        - a_hs: float or np.ndarray - The calculated Helmholtz energy.

        Reference: Equation 11 from Lafitte, 2013.
        """
        eta = self.packing_fraction(density)
        a_hs = (4 * eta - 3 * (eta ** 2)) / ((1 - eta) ** 2)
        return a_hs
    
    def compressibility_factor(self, density: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the compressibility factor.

        Parameters:
        - density: float or np.ndarray - The value(s) of the density.

        Returns:
        - k_hs: float or np.ndarray - The calculated compressibility factor.

        Reference: Equation 16 from Lafitte, 2013.
        """
        eta = self.packing_fraction(density)
        k_hs = (
            ((1 - eta) ** 4)
            /(1 + 4 * eta + 4 * (eta ** 2) - 4 * (eta ** 3) + (eta ** 4))
        )
        return k_hs
    
    def alpha(self, lambda_a: int, lambda_r: int) -> float:
        """
        Calculate alpha.

        Parameters:
        - lambda_a: int - Lambda_a parameter.
        - lambda_r: int - Lambda_r parameter.

        Returns:
        - alpha: float - The calculated alpha.

        Reference: Equation 18 from Lafitte, 2013.
        """
        C = (
            (lambda_r / (lambda_r - lambda_a))
            * ((lambda_r / lambda_a) ** (lambda_a / (lambda_r - lambda_a)))
        )
        alpha = C * ((1 / (lambda_a - 3)) - (1 / (lambda_r - 3)))
        return alpha

    def __alpha_powers(self, alpha: float) -> np.ndarray:
        """
        Calculate powers of alpha.

        Parameters:
        - alpha: float - The value of alpha.

        Returns:
        - result: np.ndarray - The calculated powers of alpha.
        """
        alpha = np.array([alpha])
        powers = np.arange(4)
        alpha_col = alpha[:, np.newaxis]
        result = alpha_col ** powers
        return result.T
    
    def __f(self, alpha: float) -> np.ndarray:
        """
        Calculate f based on alpha.

        Parameters:
        - alpha: float - The value of alpha.

        Returns:
        - f: np.ndarray - The calculated f.

        Reference: Equation 20 from Lafitte, 2013.
        """
        phi = np.array([
            [7.5365557, -359.44, 1550.9, -1.19932, -1911.28, 9236.9],
            [-37.60463, 1825.6, -5070.1, 9.063632, 21390.175, -129430],
            [71.745953, -3168, 6534.6, -17.94482, -51320.7, 357230],
            [-46.83552, 1884.2, -3288.7, 11.34027, 37064.54, -315530],
            [-2.467982, -0.82376, -2.7171, 20.52142, 1103.742, 1390.2],
            [-0.50272, -3.1935, 2.0883, -56.6377, -3264.61, -4518.2],
            [8.0956883, 3.7090, 0, 40.53683, 2556.181, 4241.6]
        ], dtype=float)
        alpha_powers = self.__alpha_powers(alpha)
        numerator = np.matmul(phi[:4, :].T, alpha_powers)
        demonimator = 1 + np.matmul(phi[4:, :].T, alpha_powers[1:])
        f = (numerator / demonimator)[:, np.newaxis]
        return f
    
    def correction_factor(self, alpha: float, packing_fraction: Union[float, np.ndarray], x0: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the correction factor.

        Parameters:
        - alpha: float - The value of alpha.
        - packing_fraction: float or np.ndarray - The value(s) of the packing fraction.
        - x0: float or np.ndarray - The value(s) of x0.

        Returns:
        - correction_factor: float or np.ndarray - The calculated correction factor.

        Reference: Equation 17 from Lafitte, 2013.
        """
        if isinstance(packing_fraction, np.ndarray):
            packing_fraction = packing_fraction[:, np.newaxis]

        f = self.__f(alpha)

        correction_factor = (
            f[0] * packing_fraction * (x0 ** 3)
            + f[1] * ((packing_fraction * (x0 ** 3)) ** 5)
            + f[2] * ((packing_fraction * (x0 ** 3)) ** 8)
        )
        return correction_factor