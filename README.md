# SAFT-VR Mie

## About this project

This repository centralizes the code used to compute perturbation thermos from the SAFT-VR Mie equation of state, this work is related to my master's of science dissertation, additional information can be found on:

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

## References
1. Lafitte T, Apostolakou A, Avendaño C, Galindo A, Adjiman CS, Müller EA, Jackson G. Accurate statistical associating fluid theory for chain molecules formed from Mie segments. J Chem Phys. 2013 Oct 21;139(15):154504. doi: 10.1063/1.4819786. PMID: 24160524.
