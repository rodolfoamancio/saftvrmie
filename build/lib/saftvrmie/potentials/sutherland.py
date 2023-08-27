import numpy as np
from typing import Union

from saftvrmie.models.carnahan_starling import CarnahanStarling

class Sutherland():
    def __init__(
        self, 
        interaction_power: float,
        segment_diameter: float,
        potential_depth: float
    ):
        self.__interaction_power = interaction_power
        self.__segment_diameter = segment_diameter
        self.__potential_depth = potential_depth

    @property
    def interaction_power(self) -> float:
        return self.__interaction_power
    
    @property
    def segment_diameter(self) -> float:
        return self.__segment_diameter
    
    @property
    def potential_depth(self) -> float:
        return self.__potential_depth
    
    def __eta_powers(self, eta: Union[float, np.ndarray]) -> np.ndarray:
        # returns an np.ndarray with shape (4, length_eta)
        if not isinstance(eta, np.ndarray):
            eta = np.array([eta])
        
        powers = np.arange(1, 5)
        alpha_col = eta[:, np.newaxis]
        result = alpha_col**powers
        return result.T

    def effective_packing_fraction(self, packing_fraction: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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
        first_order_perturbation_term = (
            -12*self.potential_depth*packing_fraction
            *(1/(self.interaction_power - 3))
            *(
                (1 - self.effective_packing_fraction(packing_fraction)/2)
                /((1 - self.effective_packing_fraction(packing_fraction))**3)
            )
        )
        return first_order_perturbation_term