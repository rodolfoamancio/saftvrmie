import numpy as np
import yaml

from typing import Union

class Reader():
    @property
    def yaml_path(self) -> str:
        return self.__yaml_path
    
    @property
    def segment_diameter(self) -> float:
        return self.__segment_diameter
    
    @property
    def potential_depth(self) -> float:
        return self.__potential_depth
    
    @property
    def repulsive_exponent(self) -> float:
        return self.__repulsive_exponent
    
    @property
    def attractive_exponent(self) -> float:
        return self.__attractive_exponent
    
    @property
    def ms(self) -> float:
        return self.__ms
    
    @property
    def molar_mass(self) -> float:
        return self.__molar_mass
    
    @property
    def temperature(self) -> Union[float, np.ndarray]:
        return self.__temperature
    
    @property
    def density(self) -> Union[float, np.ndarray]:
        return self.__density
    
    @property
    def output_filename(self) -> str:
        return self.__output_filename

    @staticmethod
    def read(yaml_path: str) -> 'Reader': 
        reader = Reader()
        reader.__yaml_path = yaml_path
        with open(yaml_path, "r") as file:
            loader = yaml.safe_load(file)
            # molecule parameters
            reader.__segment_diameter = loader["segment_diameter"]
            reader.__potential_depth = loader["potential_depth"]
            reader.__repulsive_exponent = loader["repulsive_exponent"]
            reader.__attractive_exponent = loader["attractive_exponent"]
            reader.__ms = loader["ms"]
            reader.__molar_mass = loader["molar_mass"]
            
            # simulation parameters
            reader.__temperature = loader["temperature"]
            if isinstance(reader.__temperature, list):
                reader.__temperature = np.array(loader["temperature"])
            
            reader.__density = loader["density"]
            if isinstance(reader.__density, list):
                reader.__density = np.array(loader["density"])

            # output options
            reader.__output_filename = loader["output_filename"]
        return reader