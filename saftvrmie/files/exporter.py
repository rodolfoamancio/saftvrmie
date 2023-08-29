import pandas as pd
import numpy as np
import yaml
from typing import Union

from saftvrmie.files import Reader
from saftvrmie.constants import BOLTZMANN

class Exporter():
    def export(
        self,
        reader: Reader,
        first_order_perturbation_term: Union[float, np.ndarray],
        second_order_perturbation_term: Union[float, np.ndarray],
    ) -> None:
        df_a1 = self.__get_a1_dataframe(reader, first_order_perturbation_term)
        df_a2 = self.__get_a2_dataframe(reader, second_order_perturbation_term)
        df_results = (
            pd.merge(
                df_a1,
                df_a2,
                on=["temperature", "densiry"]
            )
            .assign(
                a1_dimensionless = lambda df: df["a1"]/(reader.potential_depth*BOLTZMANN),
                a2_dimensionless = lambda df: df["a2"]/(reader.potential_depth*BOLTZMANN),
                segment_diameter = reader.segment_diameter,
                potential_depth = reader.potential_depth,
                repulsive_exponent = reader.repulsive_exponent,
                attractive_exponent = reader.attractive_exponent,
                input_filename = reader.yaml_path
            )
        )
        df_results.to_csv(f"{self.output_filename}.parquet", index=False)

    def __get_a1_dataframe(
        reader: Reader, 
        first_order_perturbation_term: Union[float, np.ndarray]
    ) -> pd.DataFrame:
        df_a1 = pd.DataFrame({
            "temperature":np.repeat(reader.temperature, first_order_perturbation_term.shape[1]),
            "density":np.tile(reader.density, first_order_perturbation_term.shape[0]),
            "a1":first_order_perturbation_term.flatten()
        })
        return df_a1
    
    def __get_a2_dataframe(
        reader: Reader, 
        second_order_perturbation_term: Union[float, np.ndarray]
    ) -> pd.DataFrame:
        df_a2 = pd.DataFrame({
            "temperature":np.repeat(reader.temperature, second_order_perturbation_term.shape[1]),
            "density":np.tile(reader.density, second_order_perturbation_term.shape[0]),
            "a2":second_order_perturbation_term.flatten()
        })
        return df_a2