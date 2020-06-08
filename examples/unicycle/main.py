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
"""Implementation of the periodic unicycle example as proposed in 

M. Zanon et al.
A Periodic Tracking MPC that is Locally Equivalent to Periodic Economic NMPC (section 7)
IFAC 2017 World Congress

:author: Jochem De Schutter

"""

import tunempc
import tunempc.pmpc as pmpc
import numpy as np
import casadi as ca
import casadi.tools as ct
import matplotlib.pyplot as plt

def problemData():

    """ Problem data, numeric constants,...
    """

    data = {}
    data['rho'] = 0.001
    data['v']   = 1

    return data

def dynamics(x, u, data, h = 1.0):

    """ System dynamics formulation (discrete time)
    """
    # state derivative
    xdot = ca.vertcat(
        data['v']*x['ez'],
        data['v']*x['ey'],
        - u['u']*x['ey'] - data['rho']*x['ez']*(x['ez']**2 + x['ey']**2 - 1.0),
        u['u']*x['ez'] - data['rho']*x['ey']*(x['ez']**2 + x['ey']**2 - 1.0) 
        )

    # set-up ode system
    f  = ca.Function('f',[x,u],[xdot])
    ode = {'x':x, 'p':u,'ode': f(x,u)}

    return [ca.integrator('F','rk',ode,{'tf':h,'number_of_finite_elements':50}), ode]

def vars():

    """ System states and controls
    """

    x = ct.struct_symMX(['z','y','ez','ey'])
    u = ct.struct_symMX(['u'])

    return x, u

def objective(x, u, data):

    # cost definition
    obj = u['u']**2 + x['z']**2 + 5*x['y']**2

    return  ca.Function('cost',[x,u],[obj])

# discretization
T = 5 # [s]
N = 30

# set-up system
x, u = vars()
nx = x.shape[0]
nu = u.shape[0]
data = problemData()

tuner = tunempc.Tuner(
    f = dynamics(x, u, data, h= T/N)[0],
    l =  objective(x,u,data),
    p = N
)

# initialization
t = np.linspace(0,T,N+1)
ez0 = list(map(np.cos, 2*np.pi*t/T))
ey0 = list(map(np.sin, 2*np.pi*t/T))
inv = list(map(lambda x, y: x**2 + y**2 - 1.0, ez0, ey0))
z0  = list(map(lambda x : data['v']*T/(2*np.pi)*np.sin(x), 2*np.pi*t/T))
y0  = list(map(lambda x : - data['v']*T/(2*np.pi)*np.cos(x), 2*np.pi*t/T))

# create initial guess
w0 = tuner.pocp.w(0.0)
for i in range(N):
    w0['x',i] = ct.vertcat(z0[i], y0[i],ez0[i], ey0[i])
w0['u'] = 2*np.pi/T

wsol = tuner.solve_ocp(w0 = w0.cat)
Hc   = tuner.convexify(solver='mosek')
S    = tuner.S

# add projection operator for terminal constraint
sys = tuner.sys

# mpc options
opts = {}
opts['p_operator'] = ca.Function(
    'p_operator',
    [sys['vars']['x']],
    [sys['vars']['x'][0:3]]
    )

ctrls = {}

# economic mpc controller
opts['ipopt_presolve'] = True
ctrls['EMPC'] = tuner.create_mpc('economic', N, opts=opts)

# normal tracking mpc controller
tuningTn = {'H': [np.diag([1.0, 1.0, 1.0, 1.0, 1.0])]*N, 'q': S['q']}
ctrls['TMPC'] = tuner.create_mpc('tracking', N, opts=opts, tuning = tuningTn)

# tuned tracking mpc controller
ctrls['TUNEMPC'] = tuner.create_mpc('tuned', N, opts=opts)

# list of controller names
ctrl_list = list(ctrls.keys())

# generate embedded solver
ACADOS_CODEGENERATE = False
if ACADOS_CODEGENERATE:

    # get system ode
    ode = ca.Function('ode',[x.cat,u.cat],[dynamics(x,u,data, h=T/N)[1]['ode']])

    # solver options
    opts = {}
    opts['qp_solver'] = 'PARTIAL_CONDENSING_HPIPM' # PARTIAL_CONDENSING_HPIPM
    opts['hessian_approx'] = 'GAUSS_NEWTON'
    opts['integrator_type'] = 'ERK'
    opts['nlp_solver_type'] = 'SQP' # SQP_RTI
    opts['qp_solver_cond_N'] = 1 # ???
    opts['print_level'] = 0
    opts['sim_method_num_steps'] = 50
    opts['tf'] = T
    opts['nlp_solver_max_iter'] = 300
    opts['nlp_solver_step_length'] = 1.0

    for ctrl_key in list(ctrls.keys()):
        _, _ = ctrls[ctrl_key].generate(ode, opts = opts, name = 'unicycle_'+ctrl_key)
        ctrl_list += [ctrl_key+'_ACADOS']

# disturbance
dist_z  = [0.1, 0.1, 0.5, 0.5]
Nstep   = [1, 34, 69, 105]

# plant simulator
plant_sim = dynamics(x, u, data, h= T/N)[0]

# initialize
x, u, l = {}, {}, {}
for ctrl_key in ctrl_list:
    x[ctrl_key] = wsol['x',0]
    u[ctrl_key] = []
    l[ctrl_key] = []

# closed-loop simulation
Nsim  = 250
tgrid = [T/N*i for i in range(Nsim)]
for k in range(Nsim):

    print('Closed-loop simulation step: {}/{}'.format(k+1,Nsim))

    # reference stage cost
    lOpt = tuner.l(wsol['x', k%N], wsol['u',k%N])

    for ctrl_key in ctrl_list:

        # add disturbance
        if k in Nstep:
            dist = dist_z[Nstep.index(k)]
            x[ctrl_key][0] += dist

        # compute feedback
        print('Compute {} feedback...'.format(ctrl_key))
        if ctrl_key[-6:] == 'ACADOS':
            u[ctrl_key].append(ctrls[ctrl_key[:-7]].step_acados(x[ctrl_key]))
        else:
            u[ctrl_key].append(ctrls[ctrl_key].step(x[ctrl_key]))
        l[ctrl_key].append(tuner.l(x[ctrl_key], u[ctrl_key][-1])-lOpt)

        # forward sim
        x[ctrl_key] = plant_sim(x0 = x[ctrl_key], p = u[ctrl_key][-1])['xf']

# plot feedback controls to check equivalence
for ctrl_key in ctrl_list:
    for i in range(nu):
        plt.figure(i)
        plt.step(tgrid,[u[ctrl_key][j][i] - wsol['u',j%N][i] for j in range(len(tgrid))])
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.grid(True)
        legend = ctrl_list
        plt.legend(legend)
        plt.title('Feedback control deviation')

plt.figure(nu)
for ctrl_key in ctrl_list:
    plt.step(tgrid,l[ctrl_key])
plt.legend(legend)
plt.autoscale(enable=True, axis='x', tight=True)
plt.grid(True)
plt.title('Stage cost deviation')

plt.show()
