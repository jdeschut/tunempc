# TuneMPC

![Build](https://github.com/jdeschut/tunempc/workflows/Build/badge.svg)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

**TuneMPC** is a Python package for economic tuning of nonlinear model predictive control (NMPC) problems.

More precisely, it implements a formal procedure that tunes a tracking (N)MPC scheme so that it is locally first-order equivalent to economic NMPC.
For user-provided system dynamics, constraints and economic objective, **TuneMPC** enables automated computation of optimal steady states and periodic trajectories, and spits out corresponding tuned stage cost matrices.

## Using TuneMPC

The interface of **TuneMPC** is based on the symbolic modeling framework [CasADi](https://web.casadi.org/).  
Input functions should be given as CasADi `Function` objects.

User-defined inputs:

```python
import casadi as ca

x = ca.MX.sym('x',<insert state dimension>) # states
u = ca.MX.sym('u',<insert control dimension>) # controls

f = ca.Function('f',[x,u],[<insert dynamics expr>],['x','u'],['xf']) # discrete system dynamics
l = ca.Function('l',[x,u],[<insert cost expr>]) # economic stage cost
h = ca.Function('h',[x,u],[<insert constraints expr>]) # constraints >= 0

p = <insert period of interest> # choose p=1 for steady-state
```


Basic application of **TuneMPC** with a few lines of code:

```python
import tunempc

tuner  = tunempc.Tuner(f, l, h, p) # construct optimal control problem
w_opt  = tuner.solve_ocp() # compute optimal p-periodic trajectory
[H, q] = tuner.convexify() # compute economically tuned stage cost matrices
```

## Installation

**TuneMPC** requires Python 3.5/3.6/3.7.

1.  Get a local copy of `tunempc`:

     ```
     git clone https://github.com/jdeschut/tunempc.git
     ```

2.   Install the package with dependencies:

     ```
     pip3 install <path_to_installation_folder>/tunempc
     ```

The following steps are optional but increase performance and robustness or enable additional features.

3. (optional) Install MOSEK SDP solver (free [academic licenses](https://www.mosek.com/products/academic-licenses/) available)

     ```
     pip3 install -f https://download.mosek.com/stable/wheel/index.html Mosek==9.0.98
     ```

4.  (optional) Install the `acados` software package for generating fast and embedded TuneMPC solvers.

     ```
     git submodule update --init --recursive
     cd external/acados
     mkdir build && cd build
     cmake -DACADOS_WITH_QPOASES=ON ..
     make install -j4 && cd ..
     pip3 install interfaces/acados_template
     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<tunempc_root>/external/acados/lib"
     ```

     More detailed instruction can be found [here](https://github.com/jdeschut/acados/blob/master/interfaces/acados_template/README.md).

5.  (optional) It is recommended to use HSL linear solvers as a plugin with IPOPT.
 In order to get the HSL solvers and render them visible to CasADi, follow these [instructions](https://github.com/casadi/casadi/wiki/Obtaining-HSL).

## Acknowledgments

This project has received funding by DFG via Research Unit FOR 2401 and by an industrial project with the company [Kiteswarms Ltd](http://www.kiteswarms.com).

## Literature

### Software

TuneMPC - A Tool for Economic Tuning of Tracking (N)MPC Problems \
J. De Schutter, M. Zanon, M. Diehl \
(pending approval)

### Theory

[A Periodic Tracking MPC that is Locally Equivalent to Periodic Economic MPC](https://www.sciencedirect.com/science/article/pii/S2405896317328987) \
M. Zanon, S. Gros, M. Diehl \
IFAC 2017 World Congress

[A tracking MPC formulation that is locally equivalent to economic MPC](https://cdn.syscop.de/publications/Zanon2016.pdf) \
M. Zanon, S. Gros, M. Diehl \
Journal of Process Control 2016

### CasADi

[CasADi - A software framework for nonlinear optimization and optimal control](http://www.optimization-online.org/DB_FILE/2018/01/6420.pdf) \
J.A.E. Andersson, J. Gillis, G. Horn, J.B. Rawlings, M. Diehl \
Mathematical Programming Computation, 2018
