# 1D Spherical FVM Solver

A 1D finite volume method (FVM) solver for spherical coordinate systems with isothermal conditions.

## Features

- 1D linear advection and Euler equations solver in spherical coordinates
- Explicit Euler or SSP RK3 time integration
- Upwind (Rusanov) and central flux schemes
- Reconstruction is 1st order

## Files

- `lin_adv.py`: Main 1D linear advection solver with 3D plotting
- `Euler.py`: Euler equations solver
- `NS.py`: Navier Stokes equations solver
- `NS_pr.py`: Navier Stokes equations solver with Peng Robinson equation of state
- `SRK.py`: Soave-Redlich-Kwong equation of state
- `PengRobinson.py`: Peng-Robinson equation of state

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Author

Sho Wada
