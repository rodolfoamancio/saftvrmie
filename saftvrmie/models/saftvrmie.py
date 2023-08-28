import numpy as np
from typing import Union

from saftvrmie.constants.constants import BOLTZMANN
from saftvrmie.potentials.mie import Mie
from saftvrmie.potentials.sutherland import Sutherland
from saftvrmie.models.carnahan_starling import CarnahanStarling

class SAFTVRMie():
    def __init__(
            self,
            attractive_power: int,
            repulsive_power: int,
            segment_diameter: float,
            potential_depth: float
        ):
        self.__attractive_power = attractive_power
        self.__repulsive_power = repulsive_power
        self.__segment_diameter = segment_diameter
        self.__potential_depth = potential_depth

        self.__mie = Mie(
            self.attractive_power,
            self.repulsive_power,
            self.segment_diameter,
            self.potential_depth
        )
        self.__C = self.mie.C

    @property
    def attractive_power(self) -> int:
        return self.__attractive_power
    
    @property
    def repulsive_power(self) -> int:
        return self.__repulsive_power

    @property
    def segment_diameter(self) -> float:
        return self.__segment_diameter
    
    @property
    def potential_depth(self) -> float:
        return self.__potential_depth

    @property
    def C(self) -> float:
        return self.__C

    @property
    def mie(self) -> Mie:
        return self.__mie

    def x0(self, diameter: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x0 = self.segment_diameter/diameter
        return x0

    def __I(self, interaction_power: int, x0: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        I = -(
            (x0**(3-interaction_power)-1)
            /(interaction_power-3)
        )
        return I
    
    def __J(self, interaction_power: int, x0: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        J = -(
            ((x0**(4-interaction_power))*(interaction_power-3) - (x0**(3-interaction_power))*(interaction_power-4) - 1)
            /((interaction_power-3)*(interaction_power-4))
        )
        return J
    
    def __B(self, interaction_power: int, packing_fraction: Union[float, np.ndarray], x0: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # return matrix with shape (length_x0, packing_fraction)
        I = self.__I(interaction_power, x0).reshape(x0.shape[0], 1)
        J = self.__J(interaction_power, x0).reshape(x0.shape[0], 1)

        B = (
            1*packing_fraction*self.potential_depth*BOLTZMANN*(
                I*(
                    (1-packing_fraction/2)
                    /((1-packing_fraction)**3)
                )
                -J*(
                    (9*packing_fraction*(1+packing_fraction))
                    /(2*((1-packing_fraction)**3))
                )
            )
        )
        return B
    
    def first_order_perturbation_term(self, beta: Union[float, np.ndarray], density: Union[float, np.ndarray]) -> Union[float, np.ndarray]:

        if not isinstance(beta, np.ndarray):
            beta = np.array([beta])

        if not isinstance(density, np.ndarray):
            density = np.array([density])

        sutherland_attractive = Sutherland(self.attractive_power, self.segment_diameter, self.potential_depth)
        sutherland_repulsive = Sutherland(self.repulsive_power, self.segment_diameter, self.potential_depth)
        carnahan_starling = CarnahanStarling(self.segment_diameter)
        
        # packing fraction
        packing_fraction = carnahan_starling.packing_fraction(density) # shape: (length_density, )

        # first order Sutherland terms
        a1S_attractive = sutherland_attractive.first_order_perturbation_term(packing_fraction)[:, np.newaxis] # shape: (length_density, 1)
        a1S_repulsive = sutherland_repulsive.first_order_perturbation_term(packing_fraction)[:, np.newaxis] # shape: (length_density, 1)

        # diameter
        diameter = self.mie.effective_diameter(beta) # shape: (length_beta, )

        # x0
        x0 = self.x0(diameter) # shape: (length_beta, )

        # B
        B_attractive = self.__B(self.attractive_power, packing_fraction, x0) # shape: (length_beta, length_density)
        B_repulsive = self.__B(self.repulsive_power, packing_fraction, x0) # shape: (length_beta, length_density)

        # # a1
        a1 = self.C*(
            ((x0[:, np.newaxis]**self.attractive_power)*(a1S_attractive + B_attractive.T).T)
            -((x0[:, np.newaxis]**self.repulsive_power)*(a1S_repulsive + B_repulsive.T).T)
        )
        return a1

    def second_order_perturbation_term(self, beta: Union[float, np.ndarray], density: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        carnahan_starling = CarnahanStarling(self.segment_diameter)
        sutherland_2a = Sutherland(2*self.attractive_power, self.segment_diameter, self.potential_depth)
        sutherland_2r = Sutherland(2*self.repulsive_power, self.segment_diameter, self.potential_depth)
        sutherland_ar = Sutherland(self.repulsive_power + self.attractive_power, self.segment_diameter, self.potential_depth)

        # diameter
        diameter = self.mie.effective_diameter(beta) # shape: (length_beta, )

        # x0
        x0 = self.x0(diameter) # shape: (length_beta, )

        # packing fraction
        packing_fraction = carnahan_starling.packing_fraction(density) # shape: (length_density, )

        # alpha
        alpha = carnahan_starling.alpha(self.attractive_power, self.repulsive_power)

        # K HS
        compressibility_factor = carnahan_starling.compressibility_factor(density)

        # correction factor
        correction_factor = carnahan_starling.correction_factor(alpha, packing_fraction, x0)

        # a1
        a1S_2a = sutherland_2a.first_order_perturbation_term(packing_fraction)
        a1S_2r = sutherland_2r.first_order_perturbation_term(packing_fraction)
        a1S_ar = sutherland_ar.first_order_perturbation_term(packing_fraction)

        # B
        B_2a = self.__B(2*self.attractive_power, packing_fraction, x0)
        B_2r = self.__B(2*self.repulsive_power, packing_fraction, x0)
        B_ar = self.__B(self.attractive_power + self.repulsive_power, packing_fraction, x0)

        # a2
        a2 = (
            (0.5*compressibility_factor*(1+correction_factor)*self.potential_depth*(self.C**2))*(
                ((x0**(2*self.attractive_power))*(a1S_2a + B_2a))
                -2*(((x0**(self.attractive_power+self.repulsive_power)))*(a1S_ar + B_ar))
                +((x0**(2*self.repulsive_power))*(a1S_2r + B_2r))
            )
        )
        return a2