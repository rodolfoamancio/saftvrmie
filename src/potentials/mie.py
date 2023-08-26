import numpy as np
from typing import Union
from constants.constants import BOLTZMANN

class Mie():
    def __init__(
            self,
            attractive_power: int,
            repulsive_power: int,
            segment_diameter: float,
            potential_depth: float
        ):
    
        self.__repulsive_power = repulsive_power
        self.__attractive_power = attractive_power
        self.__segment_diameter = segment_diameter
        self.__potential_depth = potential_depth
        self.__C = (
            (self.repulsive_power/(self.repulsive_power - self.attractive_power))
            *(
                (self.repulsive_power/self.attractive_power)
                **(self.attractive_power/(self.repulsive_power - self.attractive_power))
            )
        )

    @property
    def repulsive_power(self):
        return self.__repulsive_power

    @property
    def attractive_power(self):
        return self.__attractive_power

    @property
    def segment_diameter(self):
        return self.__segment_diameter
    
    @property
    def potential_depth(self):
        return self.__potential_depth
    
    @property
    def C(self):
        return self.__C
    
    def potential(self, distance: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        potential = self.C*self.potential_depth*BOLTZMANN*(
            ((self.segment_diameter/distance)**self.repulsive_power)
            -((self.segment_diameter/distance)**self.attractive_power)
        )
        return potential
    
    def effective_diameter(self, beta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        distance = np.linspace(0, 1, 100, endpoint=True)
        h = distance[1] - distance[0]
        potential = self.potential(distance) # shape (100,)
        if not isinstance(beta, np.ndarray):
            beta = np.array([beta])
        # beta shape (1,) or (l,)
        y = (1 - np.exp(-np.matmul(beta*potential)))
        # shape (l, 100)
        # trapezoidal rule for integration
        d = (h / 2) * (y[:, 0] + 2 * np.sum(y[:, 1:-1], axis=1) + y[:, -1])
        return d