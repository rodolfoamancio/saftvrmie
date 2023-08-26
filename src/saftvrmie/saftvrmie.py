import numpy as np
from typing import Union


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
        self.__C = (
            self.repulsive_power/(self.repulsive_power-self.attractive_power)
            *((repulsive_power/attractive_power)**(attractive_power/(repulsive_power-attractive_power)))
        )
        self.__segment_diameter = segment_diameter
        self.__potential_depth = potential_depth

    @property
    def attractive_power(self) -> int:
        return self.__attractive_power
    
    @property
    def repulsive_power(self) -> int:
        return self.__repulsive_power
    
    @property
    def C(self) -> float:
        return self.__C
    
    @property
    def segment_diameter(self) -> float:
        return self.__segment_diameter
    
    @property
    def potential_depth(self) -> float:
        return self.__potential_depth
    
    
