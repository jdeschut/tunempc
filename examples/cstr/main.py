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
#    License along with awebox; if not, write to the Free Software Foundation,
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
    f = cstr.dynamics(x, u, data),
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

# check equivalence
alpha = np.linspace(-0.1, 1.0, 2)
x0 = wsol['x',0]
dx_diehl = np.array([1.0,0.5, 100.0, 100.0]) - wsol['x',0]

# compute open-loop prediction for list of deviations in different state directions
log = []
for dist in [0]:
    dx = np.zeros((nx,1))
    dx[dist] = dx_diehl[dist]
    log.append(clt.check_equivalence(ctrls, cost, tuner.sys['h'], x0, dx, alpha))

fig_num = 1
alpha_plot = -1 # alpha value of interest
dx_plot = 0 # state direction of interest
for name in list(ctrls.keys()):

    # plot controls
    plt.figure(fig_num)
    for i in range(nu):
        plt.subplot(nu,1,i+1)
        plt.step(tgrid, [log[dx_plot][alpha_plot]['u'][name][j][i] for j in range(len(tgrid))], where='post')
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)

    # plot stage cost deviation
    plt.figure(fig_num +1)
    stage_cost_dev = [x[0] - x[1] for x in zip(log[dx_plot][alpha_plot]['l'][name],N*[lOpt])]
    plt.step(tgrid, stage_cost_dev, where='post')

    # plot state prediction
    plt.figure(fig_num +2)
    for i in range(nx):
        plt.subplot(nx,1,i+1)
        plt.plot(tgridx, [log[dx_plot][alpha_plot]['x'][name][j][i] for j in range(len(tgridx))])
        plt.plot(tgridx, [wsol['x',0][i] for j in range(len(tgridx))],  linestyle='--', color='black')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.grid(True)

# adjust control plot
plt.figure(fig_num)
plt.legend(list(ctrls.keys()))
plt.subplot(nu,1,1)
plt.title('Feedback controls')
plt.subplot(nu,1,nu)
plt.xlabel('t [s]')

# adjust stage cost plot
plt.figure(fig_num+1)
plt.legend(list(ctrls.keys()))
plt.grid(True)
plt.xlabel('t [s]')
plt.title('Stage cost deviation')
plt.autoscale(enable=True, axis='x', tight=True)

# adjust state plot
plt.figure(fig_num+2)
plt.subplot(nx,1,1)
plt.legend(list(ctrls.keys()))
plt.title('State trajectories')
plt.subplot(nx,1,nx)
plt.xlabel('t [s]')

fig_num += 3

# plot transient cost vs. alpha
plt.figure(fig_num)
transient_cost = {}
for name in list(ctrls.keys()):
    transient_cost[name] = []
    for i in range(len(alpha)):
            transient_cost[name].append(
                sum([x[0] - x[1] for x in zip(log[dx_plot][i]['l'][name],N*[lOpt])])
                )
    plt.plot(alpha, transient_cost[name], marker = 'o')

plt.grid(True)
plt.legend(list(ctrls.keys()))
plt.title('Transient cost')
plt.xlabel('alpha [-]')

plt.show()