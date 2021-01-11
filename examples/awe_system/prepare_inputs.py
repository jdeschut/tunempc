#
#    This file is part of TuneMPC.
#
#    TuneMPC -- A Tool for Economic Tuning of Tracking (N)MPC Problems.
#    Copyright (C) 2020 Jochem De Schutter, Mario Zanon, Moritz Diehl (ALU Freiburg).
#
#    TuneMPC is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 3 of the License, or (at your option) any later version.
#
#    TuneMPC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with TuneMPC; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
#!/usr/bin/python3
""" awebox optimization toolbox is available at: https://github.com/awebox/awebox
commit hash: 561f1726f28357e04b2e6a1163a1b30753670784

:author: Jochem De Schutter

"""

import awebox as awe
import awebox.tools.integrator_routines as awe_integrators
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import casadi.tools as ct
import pickle

def generate_kite_model_and_orbit(N):

    import point_mass_model

    # make default options object
    options = awe.Options(True)

    # single kite with point-mass model
    options['user_options']['system_model']['architecture'] = {1:0}
    options['user_options']['system_model']['kite_dof'] = 3
    options['user_options']['kite_standard'] = point_mass_model.data_dict()

    # trajectory should be a single drag-mode cycle
    options['user_options']['trajectory']['type'] = 'power_cycle'
    options['user_options']['trajectory']['system_type'] = 'drag_mode'

    # don't include induction effects, use simple tether drag model
    options['user_options']['induction_model'] = 'not_in_use'
    options['user_options']['tether_drag_model'] = 'split'
    options['model']['tether']['use_wound_tether'] = False

    options['nlp']['n_k'] = N

    # get point mass model data
    options = point_mass_model.set_options(options)

    # initialize and optimize trial
    trial = awe.Trial(options, 'single_kite_drag_mode')
    trial.build()
    trial.optimize(final_homotopy_step='final')
    # trial.plot(['states','controls', 'constraints'])
    # plt.show()

    # extract model data
    sol = {}
    sol['model'] = trial.generate_optimal_model()
    sol['l_t']   = trial.optimization.V_opt['xd',0,'l_t']

    # initial guess
    w_init = []
    for k in range(N):
        w_init.append(trial.optimization.V_opt['xd',k][:-3])
        w_init.append(trial.optimization.V_opt['u', k][3:6])
    sol['w0'] = ca.vertcat(*w_init)

    return sol


# discrete period of interest
N = 40
awe_sol = generate_kite_model_and_orbit(N)

# remove tether variables
model = awe_sol['model']
l_t = awe_sol['l_t']
x_shape = model['dae']['x'].shape
x_shape = (x_shape[0]-3, x_shape[1])
x = ca.MX.sym('x',*x_shape)
x_awe = ct.vertcat(x, l_t, 0.0, 0.0)

# remove fictitious forces, tether jerk...
u_shape = model['dae']['p'].shape
u_shape = (u_shape[0]-4, u_shape[1])
u = ca.MX.sym('u',*u_shape)
u_awe = ct.vertcat(0.0,0.0,0.0,u,0.0)

# remove algebraic variable
z = model['rootfinder'](0.1, x_awe, u_awe)
nx = x.shape[0]
nu = u_shape[0]
constraints = ca.vertcat(
    -model['constraints'](x_awe, u_awe,z),
    -model['var_bounds_fun'](x_awe, u_awe,z)
)

# remove redundant constraints
constraints_new = []
for i in range(constraints.shape[0]):
    if True in ca.which_depends(constraints[i],ca.vertcat(x,u)):
        constraints_new.append(constraints[i])

# create integrator
integrator = awe_integrators.rk4root(
        'F',
        model['dae'],
        model['rootfinder'],
        {'tf': 1/N, 'number_of_finite_elements':10})
xf = integrator(x0=x_awe, p=u_awe, z0 = 0.1)['xf'][:-3]
qf = integrator(x0=x_awe, p=u_awe, z0 = 0.1)['qf']

sys = {
    'f' : ca.Function('F',[x,u],[xf,qf],['x0','p'],['xf','qf']),
    'h' : ca.Function('h', [x,u], [ca.vertcat(*constraints_new)])
}

# cost function
power_output = -sys['f'](x0=x, p=u)['qf']/model['t_f']/1e3
regularization = 1/2*1e-4*ct.mtimes(u.T,u)

cost = ca.Function(
    'cost',
    [x,u],
    [power_output + regularization] #+ extra_regularization
)

# initial guess
w0 = awe_sol['w0']

# save time-continuous dynamics
xdot = ca.MX.sym('xdot', x.shape[0])
xdot_awe = ct.vertcat(xdot, 0.0, 0.0, 0.0)# remove l, ldot, lddot
z = ca.MX.sym('z', model['dae']['z']['xa'].shape[0]+model['dae']['z']['xl'].shape[0])
rm_indeces = [6,7,9] # rm l, ldot, lddot implicit ode equations
indeces = [k for k in range(nx+3+z.shape[0]) if k not in rm_indeces]
alg = model['dae']['alg'][indeces] 
alg_fun = ca.Function('alg_fun',[model['dae']['x'],model['dae']['p'],model['dae']['z']],[alg])
dyn = ca.Function(
    'dae',
    [xdot,x,u,z],
    [alg_fun(x_awe, u_awe, ct.vertcat(xdot_awe, z))],
    ['xdot','x','u','z'],
    ['dyn'])
quad_fun = ca.Function('quad_fun',[model['dae']['x'],model['dae']['p']], [model['dae']['quad']])
quad = ca.Function(
    'quad',
    [x,u],
    [quad_fun(x_awe, u_awe)],
    ['x','u'],
    ['quad']
)

# save user input info
with open('user_input.pkl','wb') as f:
        pickle.dump({
            'f': sys['f'],
            'l': cost,
            'h': sys['h'],
            'p': N,
            'w0': w0,
            'dyn': dyn,
            'quad': quad,
            'ts': model['t_f']/N
        },f)
