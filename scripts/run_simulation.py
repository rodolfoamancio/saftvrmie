import pandas as pd
import numpy as np
from argparse import ArgumentParser

from saftvrmie.files import Reader, Exporter
from saftvrmie.models import SAFTVRMie
from saftvrmie.constants import BOLTZMANN, ANGSTRON, KILOGRAM, AVOGADRO

def parse_args():
    parser = ArgumentParser(
        prog="Run simulation",
        description="Run SAFT-VR Mie simulation for perturbation terms"
    )
    parser.add_argument("filename", help="Input filename")
    args = parser.parse_args()
    return args


def run_simulation(filename):
    # Reading input
    reader = Reader.read(filename)
    output_string = (
        "Input file information\n"
        f"Attractive exponent = {reader.attractive_exponent}\n"
        f"Repulsive exponent = {reader.repulsive_exponent}\n"
        f"Segment diameter = {reader.segment_diameter} A\n"
        f"Potential depth = {reader.potential_depth} K\n"
        f"Segments per chain = {reader.ms}\n"
        f"Molar mass = {reader.molar_mass} g/mol\n\n"
        "Simulation setup\n"
        f"Temperature = {reader.temperature} K\n"
        f"Density = {reader.density} kg/m³\n\n"
        "Output\n"
        f"Output filename = {reader.output_filename}\n\n"
    )
    print(output_string)

    # Converting data
    beta = 1/BOLTZMANN*reader.temperature
    density = (
        (reader.density*KILOGRAM/(ANGSTRON**3)) # converts to g/m³
        /(reader.molar_mass) # converts to mol/m³
        *AVOGADRO # converts to molecules per m³
        *reader.ms # converts to segments per m³
    )

    # SAFT Simulation
    print("Simulation starting")
    saftvr_mie = SAFTVRMie(
        reader.attractive_exponent,
        reader.repulsive_exponent,
        reader.segment_diameter,
        reader.potential_depth
    )
    a1 = saftvr_mie.first_order_perturbation_term(beta, density)
    a2 = saftvr_mie.second_order_perturbation_term(beta, density)
    print("Perturbation terms calculated\n")

    # Exporting data
    exporter = Exporter().export(
        reader,
        a1,
        a2
    )
    print("Data exported")

if __name__ == "__main__":
    print("SAFT-VR Mie Simulation")
    
    # Parse arguments
    parser = parse_args()
    filename = parser.filename
    print("Argmuents parsed")
    print(f"Input filename {filename}\n")

    # Run simulation
    run_simulation(filename)

    print("Simulation finished")