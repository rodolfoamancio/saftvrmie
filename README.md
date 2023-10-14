# SAFT-VR Mie

## About this project

This repository centralizes the code used to compute perturbation thermos from the SAFT-VR Mie equation of state, this work is related to my master's of science dissertation, additional information can be found on:

**Important**: This code implements only the calculation of the perturbation terms ($\tilde{a}_1$, $\tilde{a}_2$) of the SAFT-VR Mie correspondent to equations 34 and 36 of the reference paper.

## Running the program

After clonin the repository, the first step is to install the `SAFTVRMie` library with. First, `cd` into the project folder and run:

```
$ pip install .
```

For running a simulation, it is necessary to execute:

```
$ python <path_to_project>/scripts/run_simulation.py <path_to_input_file>.yaml
```

Where `<path_to_input_file>.yaml` is the appropriate input file in `.yaml` extension.

The .yml file follows this structure:

```yaml
segment_diameter: 3.7412
potential_depth: 153.36
repulsive_exponent: 12.650
attractive_exponent: 6
ms: 1
molar_mass: 16.04

temperature: 600

density: 100
output_filename: c1
```

Where the parameters are:
- Mie potential parameters:
  - Segment diameter ($\sigma$) corresponding to the particle diameter in $\mathring{A}$
  - Potential depth ($\varepsilon) which is the potential depth
  - Repulsive exponent ($\lambda_r$) and attractive exponent ($\lambda_a$): the exponents for the Mie potential
- SAFT parameters:
  - Number of particles per chain ($m_s$)
- Molar mass in g/mol
- Temperature in K
- Density in kg/m³

## References
1. Lafitte T, Apostolakou A, Avendaño C, Galindo A, Adjiman CS, Müller EA, Jackson G. Accurate statistical associating fluid theory for chain molecules formed from Mie segments. J Chem Phys. 2013 Oct 21;139(15):154504. doi: 10.1063/1.4819786. PMID: 24160524.
