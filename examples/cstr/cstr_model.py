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
"""System dynamics, constraints and objective of CSTR benchmarking example found in
Chen, H., Stability and Robustness Considerations in Nonlinear Model Predictive Control, 1997. 

:author: Jochem De Schutter

"""

import numpy as np
import casadi as ca
import casadi.tools as ct

def problem_data():

    """ Problem data, numeric constants,...
    """

    # physics
    data = {}
    data['k10'] = 1.287e12
    data['k20'] = 1.287e12
    data['k30'] = 9.043e9
    data['E1']  = -9758.3
    data['E2']  = -9758.3
    data['E3']  = -8560.0
    data['DH_AB'] = 4.2
    data['DH_BC'] = -11.0
    data['DH_AD'] = -41.85
    data['rho'] = 0.9342
    data['Cp'] = 3.01
    data['kw'] = 4032.0
    data['AR'] = 0.215
    data['VR'] = 10.0
    data['mK'] = 5.0
    data['CPK'] = 2.0
    data['cA0'] = 5.10
    data['theta0'] = 104.9

    # tuning
    data['rhoJ'] = 0.0        # regularization weight
    data['num_steps'] = 20    # number_of_finite_elements in integrator
    data['ts'] = 20           # sampling time
    data['integrator'] = 'rk' # numerical integrator

    return data

def arrhenius(x,u,data,i):

    k = data['k{}0'.format(i)]*np.exp(
            data['E{}'.format(i)]/(x['theta']+273.15)
        )

    return k

def dynamics(x, u, data):

    """ System dynamics function (discrete time)
    """

    # state derivative expression
    xdot = ca.vertcat(
        u['Vdot']*(data['cA0'] - x['cA']) - arrhenius(x,u,data,1)*x['cA'] - arrhenius(x,u,data,3)*x['cA']*x['cA'],
        -u['Vdot']*x['cB'] + arrhenius(x,u,data,1)*x['cA'] - arrhenius(x,u,data,2)*x['cB'],
        u['Vdot']*(data['theta0']-x['theta']) - 1.0/(data['rho']*data['Cp'])*(
            arrhenius(x,u,data,1)*x['cA']*data['DH_AB'] + 
            arrhenius(x,u,data,2)*x['cB']*data['DH_BC'] +
            arrhenius(x,u,data,3)*x['cA']*x['cA']*data['DH_AD']
        ) + data['kw']*data['AR']/(data['rho']*data['Cp']*data['VR'])*(x['thetaK']- x['theta']),
        1.0/(data['mK']*data['CPK'])*(u['QdotK']+data['kw']*data['AR']*(x['theta']-x['thetaK']))
        )

    # create ode for integrator
    ode = {'x':x, 'p':u,'ode': xdot/3600}

    # integrator
    F = ca.integrator('F',data['integrator'],ode,{'tf':data['ts'],'number_of_finite_elements':data['num_steps']})

    return [F, ode]

def vars():

    """ System states and controls
    """

    x = ct.struct_symMX(['cA','cB','theta', 'thetaK'])
    u = ct.struct_symMX(['Vdot','QdotK'])

    return x, u

def objective(x, u, data):

    """ Economic objective function
    """
    
    # cost definition
    e_obj = - x['cB']/data['cA0']

    # regularization
    reg = data['rhoJ']*(
        0*(x['cA'] - 2.1402)**2 + \
        0*(x['cB'] - 1.0903)**2 + \
        0*(x['theta'] - 114.191)**2 + \
        0*(x['thetaK'] -  112.9066 )**2 + \
        1e-4*(u['Vdot']-14.19)**2 + \
        1e-4*(u['QdotK']-(-1113.5))**2
        )

    return ca.Function('economic_cost',[x,u],[100*(e_obj+reg)]), ca.Function('cost_comp', [x,u], [e_obj, reg])


def tracking_cost(nw):

    """ Tracking cost function
    """

    # reference parameters
    w =  ca.MX.sym('w', (nw, 1))
    wref = ca.MX.sym('wref', (nw, 1))
    H = ca.MX.sym('H', (nw, nw))
    q = ca.MX.sym('H', (nw, 1))

    # cost definition
    dw   = w - wref
    obj  = 0.5*ct.mtimes(dw.T, ct.mtimes(H, dw)) + ct.mtimes(q.T,dw)

    return ca.Function('tracking_cost',[w, wref, H, q],[obj])

def constraints(x, u, data):
    
    """ Path inequality constraints function (convention h(x,u) >= 0)
    """

    constr = ca.vertcat(
        u['Vdot'] - 5,
        35.0 - u['Vdot'],
        u['QdotK'] + 9000.0,
        -u['QdotK']
    )

    return ca.Function('h', [x,u], [constr])

def initial_guess():

    """ Initial guess for economic steady state optimization
    """

    return ca.vertcat(2.1402, 1.0903, 114.191, 112.9066, 14.19, -1113.5)