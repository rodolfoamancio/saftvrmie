import numpy as np
from typing import Union, List

from saftvrmie.constants.constants import BOLTZMANN

class Mie():
    """
    Class for Mie potential calculations.
    """
    def __init__(
            self,
            attractive_exponent: float,
            repulsive_exponent: float,
            segment_diameter: float,
            potential_depth: float
        ):
        """
        Initialize the Mie potential parameters.

        Parameters:
        - attractive_exponent: int - The exponent for the attractive power term in the Mie potential equation.
        - repulsive_exponent: int - The exponent for the repulsive power term in the Mie potential equation.
        - segment_diameter: float - The diameter of the segment in the Mie potential equation.
        - potential_depth: float - The depth of the potential well in the Mie potential equation.
        """
        
        self.__repulsive_power = repulsive_exponent
        self.__attractive_power = attractive_exponent
        self.__segment_diameter = segment_diameter
        self.__potential_depth = potential_depth
        self.__C = (
            (self.repulsive_power / (self.repulsive_power - self.attractive_power))
            * (
                (self.repulsive_power / self.attractive_power)
                ** (self.attractive_power / (self.repulsive_power - self.attractive_power))
            )
        )

    @property
    def repulsive_power(self) -> float:
        """
        Get the repulsive power.
        """
        return self.__repulsive_power

    @property
    def attractive_power(self) -> float:
        """
        Get the attractive power.
        """
        return self.__attractive_power

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
    
    @property
    def C(self):
        """
        Get the constant C in the Mie potential equation.
        """
        return self.__C
    
    def potential(self, distance: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the Mie potential for a given distance.

        Parameters:
        - distance: float or np.ndarray - The distance(s) at which to calculate the potential.

        Returns:
        - potential: float or np.ndarray - The calculated potential(s).
        """
        BOLTZMANN = 1.380649e-23 # Boltzmann constant
        
        potential = self.C * self.potential_depth * BOLTZMANN * (
            ((self.segment_diameter / distance) ** self.repulsive_power)
            - ((self.segment_diameter / distance) ** self.attractive_power)
        )
        return potential
    
    def __aux_function(self, beta: Union[float, np.ndarray], distance: np.ndarray) -> np.ndarray:
        """
        Auxiliary function for calculating effective diameter. Not meant to be used directly.

        Parameters:
        - beta: float or np.ndarray - The value(s) of beta.
        - distance: np.ndarray - The distances at which to calculate the auxiliary function.

        Returns:
        - aux_function: np.ndarray - The calculated auxiliary function.
        """
        if not isinstance(beta, np.ndarray):
            beta = np.array([beta])
        beta = beta[:, np.newaxis]
        aux_function = 1 - np.exp(-1 * np.matmul(beta, self.potential(distance).T))
        aux_function = np.where(np.isnan(aux_function), 1, aux_function)
        return aux_function
    
    def effective_diameter(self, beta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the effective diameter using the Mie potential.

        Parameters:
        - beta: float or np.ndarray - The value(s) of beta.

        Returns:
        - d: float or np.ndarray - The calculated effective diameter(s).
        """
        distance = np.linspace(0, self.segment_diameter, 100, endpoint=True)[:, np.newaxis]
        distance = distance + 1E-10
        h = distance[1] - distance[0]
        y = self.__aux_function(beta, distance)
        d = (h / 2) * (y[:, 0] + 2 * np.sum(y[:, 1:-1], axis=1) + y[:, -1])
        return d
