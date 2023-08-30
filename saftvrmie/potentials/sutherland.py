import numpy as np
from typing import Union

from saftvrmie.constants import BOLTZMANN

class Sutherland():
    """
    Class for Sutherland potential calculations.
    """
    def __init__(
        self, 
        interaction_exponent: float,
        segment_diameter: float,
        potential_depth: float
    ):
        """
        Initialize the Sutherland potential parameters.

        Parameters:
        - interaction_exponent: float - The exponent for the interaction power term in the Sutherland potential equation.
        - segment_diameter: float - The diameter of the segment in the Sutherland potential equation.
        - potential_depth: float - The depth of the potential well in the Sutherland potential equation.
        """
        self.__interaction_power = interaction_exponent
        self.__segment_diameter = segment_diameter
        self.__potential_depth = potential_depth

    @property
    def interaction_power(self) -> float:
        """
        Get the interaction power.
        """
        return self.__interaction_power
    
    @property
    def segment_diameter(self) -> float:
        """
        Get the segment diameter.
        """
        return self.__segment_diameter
    
    @property
    def potential_depth(self) -> float:
        """
        Get the potential depth.
        """
        return self.__potential_depth
    
    def __eta_powers(self, eta: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate powers of eta.

        Parameters:
        - eta: float or np.ndarray - The value(s) of eta.

        Returns:
        - result: np.ndarray - The calculated powers of eta.
        """
        if not isinstance(eta, np.ndarray):
            eta = np.array([eta])
        
        powers = np.arange(1, 5)
        eta = eta[:, np.newaxis]
        result = eta ** powers
        return result.T

    def effective_packing_fraction(self, packing_fraction: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the effective packing fraction.

        Parameters:
        - packing_fraction: float or np.ndarray - The value(s) of the packing fraction.

        Returns:
        - effective_packing_fraction: float or np.ndarray - The calculated effective packing fraction.
        """
        parameters = np.array([
            [0.81096, 1.7888, -37.578, 92.284],
            [1.0505, -19.341, 151.26, -465.50],
            [-1.9057, 22.845, -228.14, 973.92],
            [1.0885, -6.1962, 106.98, -677.64]
        ])
        lambda_array = np.array([1, 1/self.interaction_power, 1/(self.interaction_power**2), 1/(self.interaction_power**3)])
        c = np.matmul(parameters, lambda_array)
        eta_powers = self.__eta_powers(packing_fraction)
        effective_packing_fraction = np.matmul(c.T, eta_powers).T
        return effective_packing_fraction
    
    def first_order_perturbation_term(self, packing_fraction: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the first-order perturbation term.

        Parameters:
        - packing_fraction: float or np.ndarray - The value(s) of the packing fraction.

        Returns:
        - first_order_perturbation_term: float or np.ndarray - The calculated first-order perturbation term.
        """
        first_order_perturbation_term = (
            -12 * self.potential_depth * BOLTZMANN * packing_fraction *
            (1 / (self.interaction_power - 3)) *
            ( (1 - self.effective_packing_fraction(packing_fraction) / 2) /
              ((1 - self.effective_packing_fraction(packing_fraction)) ** 3)
            )
        )
        return first_order_perturbation_term