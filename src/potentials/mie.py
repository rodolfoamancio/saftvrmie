

class Mie():
    def __init__(
            self,
            repulsive_power: int,
            attractive_power: int,
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