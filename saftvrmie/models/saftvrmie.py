import numpy as np
from typing import Union

from saftvrmie.constants import BOLTZMANN
from saftvrmie.potentials import Mie, Sutherland
from saftvrmie.models import CarnahanStarling

class SAFTVRMie():
    """
    SAFT-VR Mie class
    Based on Lafitte, Thomas, et al. "Accurate statistical associating fluid theory 
    for chain molecules formed from Mie segments." The Journal of chemical physics 139.15 (2013).
    """
    def __init__(
            self,
            attractive_exponent: float,
            repulsive_exponent: float,
            segment_diameter: float,
            potential_depth: float
        ):
        """
        Initializes the SAFT-VR Mie class

        Parameters:
        - attractive_exponent: int - The exponent for the attractive power term in the Mie potential equation.
        - repulsive_exponent: int - The exponent for the repulsive power term in the Mie potential equation.
        - segment_diameter: float - The diameter of the segment in the Mie potential equation.
        - potential_depth: float - The depth of the potential well in the Mie potential equation.
        """
        self.__attractive_exponent = attractive_exponent
        self.__repulsive_exponent = repulsive_exponent
        self.__segment_diameter = segment_diameter
        self.__potential_depth = potential_depth

        self.__mie = Mie(
            self.attractive_exponent,
            self.repulsive_exponent,
            self.segment_diameter,
            self.potential_depth
        )
        self.__C = self.mie.C

    @property
    def attractive_exponent(self) -> float:
        """
        Get the attractive power.
        """
        return self.__attractive_exponent
    
    @property
    def repulsive_exponent(self) -> float:
        """
        Get the repulsive power.
        """
        return self.__repulsive_exponent

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
    def C(self) -> float:
        """
        Get the constant C in the Mie potential equation.
        """
        return self.__C

    @property
    def mie(self) -> Mie:
        """
        Get the corresponding Mie potential object from the SAFT-VR Mie EoS
        """
        return self.__mie

    def x0(self, diameter: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Computes the reduced diameter

        Parameters:
        - diameter: float or np.ndarray - The input diameter

        Returns:
        - x0: the reduced diameter

        Reference: Equation 22 details from Lafitte, 2013.
        """
        x0 = self.segment_diameter/diameter
        return x0

    def __I(self, interaction_exponent: int, x0: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Computes the term I from the SAFT-VR Mie EoS. 

        Parameters:
        - interaction_exponent: int - the interaction exponent (lambda) for the potential
        - x0: float or np.ndarray - the reduced distance(s)

        Retunrs:
        - I: float or np.ndarray - the I term

        Reference: Equation 28 from Lafitte, 2013.
        """
        I = -(
            (x0**(3-interaction_exponent)-1)
            /(interaction_exponent-3)
        )
        return I
    
    def __J(self, interaction_exponent: int, x0: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Computes the term I from the SAFT-VR Mie EoS. 

        Parameters:
        - interaction_exponent: int - the interaction exponent (lambda) for the potential
        - x0: float or np.ndarray - the reduced distance(s)

        Retunrs:
        - I: float or np.ndarray - the I term

        Reference: Equation 29 from Lafitte, 2013.
        """
        J = -(
            ((x0**(4-interaction_exponent))*(interaction_exponent-3) - (x0**(3-interaction_exponent))*(interaction_exponent-4) - 1)
            /((interaction_exponent-3)*(interaction_exponent-4))
        )
        return J
    
    def __B(self, interaction_exponent: int, packing_fraction: Union[float, np.ndarray], x0: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Computes the term B from the SAFT-VR Mie EoS. 

        Parameters:
        - interaction_exponent: int - the interaction exponent (lambda) for the potential
        - packing_fraction: float or np.ndarray - the packing fraction value(s) of the system
        - x0: float or np.ndarray - the reduced distance(s)

        Retunrs:
        - B: float or np.ndarray - the I term

        Reference: Equation 33 from Lafitte, 2013.
        """
        I = self.__I(interaction_exponent, x0).reshape(x0.shape[0], 1)
        J = self.__J(interaction_exponent, x0).reshape(x0.shape[0], 1)

        B = (
            12*packing_fraction*self.potential_depth*BOLTZMANN*(
                I*(
                    (1-packing_fraction/2)
                    /((1-packing_fraction)**3)
                )
                -J*(
                    (9*packing_fraction*(1+packing_fraction))
                    /(2*((1-packing_fraction)**3))
                )
            )
        ).T
        return B
    
    def first_order_perturbation_term(self, beta: Union[float, np.ndarray], density: Union[float, np.ndarray]) -> np.ndarray:
        """
        Computes the first order perturbation term

        Parameters:
        - beta: float or np.ndarray - beta value(s) for calculating the perturbation terms
        - density: float or np.ndarray - density value(s) in segments per A³ for calculating perturbation terms

        Returns:
        - a1: float - the pertubation term in J

        Reference: Equation 34 from Lafitte, 2013.
        """
        if not isinstance(beta, np.ndarray):
            beta = np.array([beta])

        if not isinstance(density, np.ndarray):
            density = np.array([density])

        sutherland_attractive = Sutherland(self.attractive_exponent, self.segment_diameter, self.potential_depth)
        sutherland_repulsive = Sutherland(self.repulsive_exponent, self.segment_diameter, self.potential_depth)
        carnahan_starling = CarnahanStarling(self.segment_diameter)
        
        # packing fraction
        packing_fraction = carnahan_starling.packing_fraction(density) # shape: (length_density, )

        # first order Sutherland terms
        a1S_a = sutherland_attractive.first_order_perturbation_term(packing_fraction)[:, np.newaxis] # shape: (length_density, 1)
        a1S_r = sutherland_repulsive.first_order_perturbation_term(packing_fraction)[:, np.newaxis] # shape: (length_density, 1)

        # diameter
        diameter = self.mie.effective_diameter(beta) # shape: (length_beta, )

        # x0
        x0 = self.x0(diameter) # shape: (length_beta, )

        # B
        B_a = self.__B(self.attractive_exponent, packing_fraction, x0) # shape: (length_beta, length_density)
        B_r = self.__B(self.repulsive_exponent, packing_fraction, x0) # shape: (length_beta, length_density)

        # # a1
        a1 = self.C*(
            ((x0[:, np.newaxis]**self.attractive_exponent)*(a1S_a + B_a).T)
            -((x0[:, np.newaxis]**self.repulsive_exponent)*(a1S_r + B_r).T)
        )
        return a1

    def second_order_perturbation_term(self, beta: Union[float, np.ndarray], density: Union[float, np.ndarray]) -> np.ndarray:
        """
        Computes the second order perturbation term

        Parameters:
        - beta: float or np.ndarray - beta value(s) for calculating the perturbation terms
        - density: float or np.ndarray - density value(s) in segments per A³ for calculating perturbation terms

        Returns:
        - a2: float - the pertubation term in J

        Reference: Equation 36 from Lafitte, 2013.
        """
        if not isinstance(beta, np.ndarray):
            beta = np.array([beta])

        if not isinstance(density, np.ndarray):
            density = np.array([density])
        
        carnahan_starling = CarnahanStarling(self.segment_diameter)
        sutherland_2a = Sutherland(2*self.attractive_exponent, self.segment_diameter, self.potential_depth)
        sutherland_2r = Sutherland(2*self.repulsive_exponent, self.segment_diameter, self.potential_depth)
        sutherland_ar = Sutherland(self.repulsive_exponent + self.attractive_exponent, self.segment_diameter, self.potential_depth)

        # diameter
        diameter = self.mie.effective_diameter(beta) # shape: (length_beta, )

        # x0
        x0 = self.x0(diameter) # shape: (length_beta, )

        # packing fraction
        packing_fraction = carnahan_starling.packing_fraction(density) # shape: (length_density, )

        # alpha
        alpha = carnahan_starling.alpha(self.attractive_exponent, self.repulsive_exponent)

        # K HS
        compressibility_factor = carnahan_starling.compressibility_factor(density)[:, np.newaxis] # shape: (length_density, 1)

        # correction factor
        correction_factor = carnahan_starling.correction_factor(alpha, packing_fraction, x0) # shape: (length_density, lenght_beta)

        # a1
        a1S_2a = sutherland_2a.first_order_perturbation_term(packing_fraction)[:, np.newaxis] # shape: (length_density, 1) 
        a1S_2r = sutherland_2r.first_order_perturbation_term(packing_fraction)[:, np.newaxis] # shape: (length_density, 1)
        a1S_ar = sutherland_ar.first_order_perturbation_term(packing_fraction)[:, np.newaxis] # shape: (length_density, 1)

        # B
        B_2a = self.__B(2*self.attractive_exponent, packing_fraction, x0) # shape: (length_density, length_beta)
        B_2r = self.__B(2*self.repulsive_exponent, packing_fraction, x0) # shape: (length_density, length_beta)
        B_ar = self.__B(self.attractive_exponent + self.repulsive_exponent, packing_fraction, x0) # shape: (length_density, length_beta)

        # a2
        a2 = (
            (
                0.5
                *compressibility_factor
                *(1+correction_factor)
                *(self.potential_depth*BOLTZMANN)
                *(self.C**2)
            )
            *(
                ((x0**(2*self.attractive_exponent))*(a1S_2a + B_2a))
                -2*(((x0**(self.attractive_exponent+self.repulsive_exponent)))*(a1S_ar + B_ar))
                +((x0**(2*self.repulsive_exponent))*(a1S_2r + B_2r))
            )
        ).T
        return a2