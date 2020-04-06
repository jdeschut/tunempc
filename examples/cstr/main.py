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
"""Implementation of tuned steady-state tracking NMPC for the CSTR
benchmarking example found in:

Chen, H. et al.,
Nonlinear Predictive Control of a Benchmark CSTR
Proc. 3rd European Control Conference ECC’95, Rome/Italy

:author: Jochem De Schutter

"""

import tunempc
import tunempc.pmpc as pmpc
import tunempc.closed_loop_tools as clt
import cstr_model as cstr
import numpy as np
import casadi as ca
import casadi.tools as ct
import matplotlib.pyplot as plt
from statistics import mean

# set-up system
x, u = cstr.vars()
data = cstr.problem_data()
h    = cstr.constraints(x, u, data)
nx = x.shape[0]
nu = u.shape[0]

# simulation parameters
Ts = 20 # sampling time
Nsim = 70 # closed loop num of simulation steps
N = 60 # nmpc horizon length

# integrator options
data['rhoJ'] = 1e-1       # regularization
data['num_steps'] = 20    # number_of_finite_elements in integrator
data['ts'] = Ts           # sampling time
data['integrator'] = 'rk' # numerical integrator

# cost function
cost, cost_comp = cstr.objective(x,u,data)

# solve steady state problem
tuner = tunempc.Tuner(
    f = cstr.dynamics(x, u, data)[0],
    l = cost,
    h = h,
    p = 1
)
wsol = tuner.solve_ocp(w0 = cstr.initial_guess())
lOpt = cost(wsol['x',0], wsol['u',0])
Hc   = tuner.convexify()
S    = tuner.S

# prediction time grid
tgrid = [Ts*k for k in range(N)]
tgridx = [Ts*k for k in range(N+1)]

# create controllers
ctrls = {}

# economic mpc controller
opts = {'ipopt_presolve': False, 'slack_flag': 'active'}
ctrls['economic'] = tuner.create_mpc('economic',N, opts=opts)

# normal tracking mpc controller
tuningTn = {'H': [np.diag([0.2, 1.0, 0.5, 0.2, 0.5, 0.5])], 'q': S['q']}
ctrls['tracking'] = tuner.create_mpc('tracking', N, opts=opts, tuning=tuningTn)

# tuned tracking mpc controller
ctrls['tuned'] = tuner.create_mpc('tuned', N, opts=opts)

ACADOS_CODEGENERATE = False
if ACADOS_CODEGENERATE:

    # get system ode
    ode = ca.Function('ode',[x.cat,u.cat],[cstr.dynamics(x,u,data)[1]['ode']])

    # solver options
    opts = {}
    opts['qp_solver'] = 'FULL_CONDENSING_QPOASES' # PARTIAL_CONDENSING_HPIPM
    opts['hessian_approx'] = 'GAUSS_NEWTON'
    opts['integrator_type'] = 'ERK'
    opts['nlp_solver_type'] = 'SQP' # SQP_RTI
    opts['qp_solver_cond_N'] = 1 # ???
    opts['print_level'] = 0
    opts['sim_method_num_steps'] = data['num_steps']
    opts['tf'] = N*data['ts']
    opts['nlp_solver_max_iter'] = 300
    opts['nlp_solver_step_length'] = 1.0

    acados_ocp_solver, acados_integrator = ctrls['tuned'].generate(
        ode, opts = opts, name = 'unicycle'
        )

    ctrls_acados = {'tuned_acados': ctrls['tuned']}

# check equivalence
alpha = np.linspace(-0.1, 1.0, 2)
x0 = wsol['x',0]
dx_diehl = np.array([1.0,0.5, 100.0, 100.0]) - wsol['x',0]

# compute open-loop prediction for list of deviations in different state directions
log = []
log_acados = []
for dist in [0]:
    dx = np.zeros((nx,1))
    dx[dist] = dx_diehl[dist]
    log.append(clt.check_equivalence(ctrls, cost, tuner.sys['h'], x0, dx, alpha))

    if ACADOS_CODEGENERATE:
        log_acados.append(clt.check_equivalence(ctrls_acados, cost, tuner.sys['h'], x0, dx, alpha, flag='acados'))

fig_num = 1
alpha_plot = -1 # alpha value of interest
dx_plot = 0 # state direction of interest
ctrl_list = list(ctrls.keys())
if ACADOS_CODEGENERATE:
    ctrl_list += list(ctrls_acados.keys())
for name in ctrl_list:

    if name[-6:] != 'acados':
        logg = log[dx_plot][alpha_plot]
        ls = '-'
    else:
        logg = log_acados[dx_plot][alpha_plot]
        ls = '--'

    # plot controls
    plt.figure(fig_num)
    for i in range(nu):
        plt.subplot(nu,1,i+1)
        plt.step(tgrid, [logg['u'][name][j][i] for j in range(len(tgrid))], where='post', linestyle = ls)
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)

    # plot stage cost deviation
    plt.figure(fig_num +1)
    stage_cost_dev = [x[0] - x[1] for x in zip(logg['l'][name],N*[lOpt])]
    plt.step(tgrid, stage_cost_dev, where='post', linestyle = ls)

    # plot state prediction
    plt.figure(fig_num +2)
    for i in range(nx):
        plt.subplot(nx,1,i+1)
        plt.plot(tgridx, [logg['x'][name][j][i] for j in range(len(tgridx))], linestyle = ls)
        plt.plot(
            tgridx,
            [wsol['x',0][i] for j in range(len(tgridx))],
            linestyle='--',
            color='black',
            label='_nolegend_')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.grid(True)

# adjust control plot
plt.figure(fig_num)
plt.legend(ctrl_list)
plt.subplot(nu,1,1)
plt.title('Feedback controls')
plt.subplot(nu,1,nu)
plt.xlabel('t [s]')

# adjust stage cost plot
plt.figure(fig_num+1)
plt.legend(ctrl_list)
plt.grid(True)
plt.xlabel('t [s]')
plt.title('Stage cost deviation')
plt.autoscale(enable=True, axis='x', tight=True)

# adjust state plot
plt.figure(fig_num+2)
plt.subplot(nx,1,1)
plt.legend(ctrl_list)
plt.title('State trajectories')
plt.subplot(nx,1,nx)
plt.xlabel('t [s]')

fig_num += 3

# plot transient cost vs. alpha
plt.figure(fig_num)
transient_cost = {}
for name in ctrl_list:

    if name[-6:] != 'acados':
        logg = log[dx_plot]
        ls = '-'
    else:
        logg = log_acados[dx_plot]
        ls = '--'

    transient_cost[name] = []
    for i in range(len(alpha)):
            transient_cost[name].append(
                sum([x[0] - x[1] for x in zip(logg[i]['l'][name],N*[lOpt])])
                )
    plt.plot(alpha, transient_cost[name], marker = 'o', linestyle = ls)

plt.grid(True)
plt.legend(ctrl_list)
plt.title('Transient cost')
plt.xlabel('alpha [-]')

plt.show()