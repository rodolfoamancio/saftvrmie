import pandas as pd
import numpy as np
from typing import Union

from saftvrmie.files import Reader
from saftvrmie.constants import BOLTZMANN, ANGSTRON, KILOGRAM, AVOGADRO

class Exporter():
    """
    Class for exporting data
    """
    @staticmethod
    def export(
        reader: Reader,
        first_order_perturbation_term: Union[float, np.ndarray],
        second_order_perturbation_term: Union[float, np.ndarray],
    ) -> None:
        """
        Method for exporting data to csv

        Parameters:
        - reader: Reader - The reader class with simulation parameters
        - first_order_perturbation_term: Union[float, np.ndarray] - First order perturbation terms results
        - second_order_perturbation_term: Union[float, np.ndarray] - Second order perturbation terms results

        Returns:
        None
        """
        df_a1 = pd.DataFrame({
            "temperature":np.repeat(reader.temperature, first_order_perturbation_term.shape[1]),
            "density":np.tile(reader.density, first_order_perturbation_term.shape[0]),
            "a1":first_order_perturbation_term.flatten()
        })
        df_a2 = pd.DataFrame({
            "temperature":np.repeat(reader.temperature, second_order_perturbation_term.shape[1]),
            "density":np.tile(reader.density, second_order_perturbation_term.shape[0]),
            "a2":second_order_perturbation_term.flatten()
        })
        df_results = (
            pd.merge(
                df_a1,
                df_a2,
                on=["temperature", "density"]
            )
            .assign(
                a1_dimensionless = lambda df: df["a1"]/(reader.potential_depth*BOLTZMANN),
                a2_dimensionless = lambda df: df["a2"]/((reader.potential_depth*BOLTZMANN)**2),
                segment_diameter = reader.segment_diameter,
                potential_depth = reader.potential_depth,
                repulsive_exponent = reader.repulsive_exponent,
                attractive_exponent = reader.attractive_exponent,
                ms = reader.ms,
                input_filename = reader.yaml_path,
                temperature_dimensionless = lambda df: df["temperature"]/reader.potential_depth,
                density_dimensionless = lambda df: (
                    df["density"] # density in kg/m³
                    *KILOGRAM # g/m³
                    *(1/reader.molar_mass) # mol/m³
                    *AVOGADRO # molecules/m³
                    *reader.ms # segments/m³
                    *((reader.segment_diameter*ANGSTRON)**3) # dimensionless unit
                )
            )
        )
        df_results.to_csv(f"{reader.output_filename}.csv", index=False)