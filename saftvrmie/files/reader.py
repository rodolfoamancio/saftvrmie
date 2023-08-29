import numpy as np
import yaml

from typing import Union, Self

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
    def temperature(self) -> Union[float, np.ndarray]:
        return self.__temperature
    
    @property
    def density(self) -> Union[float, np.ndarray]:
        return self.__density
    
    @property
    def output_filename(self) -> str:
        return self.__output_filename

    @classmethod
    def read(self, yaml_path: str) -> Self: 
        self.__yaml_path = yaml_path
        with yaml.load(self.__yaml_path, Loader=yaml.SafeLoader()) as loader:
            # molecule parameters
            self.__segment_diameter = loader["segment_diameter"]
            self.__potential_depth = loader["potential_depth"]
            self.__repulsive_exponent = loader["repulsive_exponent"]
            self.__attractive_exponent = loader["attractive_exponent"]
            
            # simulation parameters
            self.__temperature = loader["temperature"]
            if isinstance(self.__temperature, list):
                self.__temperature = np.array(loader["temperature"])
            
            self.__density = loader["density"]
            if isinstance(self.__density, list):
                self.__density = np.array(loader["density"])

            # output options
            self.__output_filename = loader["output_filename"]
        return self