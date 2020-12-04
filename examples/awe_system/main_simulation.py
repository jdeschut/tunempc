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
""" Example of a periodic, single-aircraft drag-mode airborne wind energy system.

The user inputs are generated by the file "prepare_inputs.py"

The reference trajectory and convexified sensitivities are computed in "main.py".

Example description found in:

TuneMPC - A Tool for Economic Tuning of Tracking (N)MPC Problems
J. De Schutter, M. Zanon, M. Diehl
IEEE Control Systems Letters 2020

:author: Jochem De Schutter

"""

import tunempc.pmpc as pmpc
import tunempc.preprocessing as preprocessing
import tunempc.mtools as mtools
import tunempc.closed_loop_tools as clt
import casadi as ca
import casadi.tools as ct
import numpy as np
import pickle
import collections
import copy
import matplotlib.pyplot as plt

# load user input
with open('user_input.pkl','rb') as f:
    user_input = pickle.load(f)

# load convexified reference
with open('convex_reference.pkl','rb') as f:
    sol = pickle.load(f)

# add variables to sys again
vars = collections.OrderedDict()
for var in ['x','u','us']:
    vars[var] = ca.MX.sym(var, sol['sys']['vars'][var])
sol['sys']['vars'] = vars
nx = sol['sys']['vars']['x'].shape[0]
nu = sol['sys']['vars']['u'].shape[0]
ns = sol['sys']['vars']['us'].shape[0]

# set-up open-loop scenario
Nmpc  = 20
alpha_steps = 20
TUNEMPC_SIM = False # simulate with TuneMPC built-in MPC controllers
ACADOS_SIM = True # simulate with acados code-generated MPC solvers

# tether length
l_t = np.sqrt(
    sol['wsol']['x',0][0]**2 +
    sol['wsol']['x',0][1]**2 +
    sol['wsol']['x',0][2]**2
)

opts = {}
# add projection operator for terminal constraint
opts['p_operator'] = ca.Function(
    'p_operator',
    [sol['sys']['vars']['x']],
    [ct.vertcat(sol['sys']['vars']['x'][1:3],
    sol['sys']['vars']['x'][4:])]
)

# add MPC slacks to active constraints
mpc_sys = preprocessing.add_mpc_slacks(
    sol['sys'],
    sol['lam_g'],
    sol['indeces_As'],
    slack_flag = 'active'
)
# create controllers
ctrls = {}

# # economic MPC
tuning = {'H': sol['S']['Hc'], 'q': sol['S']['q']}
ctrls['EMPC'] = pmpc.Pmpc(
    N = Nmpc,
    sys = mpc_sys,
    cost = user_input['l'],
    wref = sol['wsol'],
    lam_g_ref = sol['lam_g'],
    sensitivities= sol['S'],
    options = opts
)

# prepare tracking cost and initialization
tracking_cost = mtools.tracking_cost(nx+nu+ns)
lam_g0 = copy.deepcopy(sol['lam_g'])
lam_g0['dyn'] = 0.0
lam_g0['g'] = 0.0
  
# standard tracking MPC
tuningTn = {'H': [np.diag((nx+nu)*[1]+ns*[1e-10])]*user_input['p'], 'q': sol['S']['q']}
ctrls['TMPC_1'] = pmpc.Pmpc(
    N = Nmpc,
    sys = mpc_sys,
    cost = tracking_cost,
    wref = sol['wsol'],
    tuning = tuningTn,
    lam_g_ref = lam_g0,
    sensitivities= sol['S'],
    options = opts
)
# manually tuned tracking MPC
Ht2 = [np.diag([0.1,0.1,0.1, 1.0, 1.0, 1.0, 1.0e3, 1.0, 100.0, 1.0, 1.0, 1.0, 1e-10, 1e-10, 1e-10])]*user_input['p']
tuningTn2 = {'H': Ht2, 'q': sol['S']['q']}
ctrls['TMPC_2'] = pmpc.Pmpc(
    N = Nmpc,
    sys = mpc_sys,
    cost = tracking_cost,
    wref = sol['wsol'],
    tuning = tuningTn2,
    lam_g_ref = lam_g0,
    sensitivities= sol['S'],
    options = opts
)

# tuned tracking MPC
ctrls['TUNEMPC'] = pmpc.Pmpc(
    N = Nmpc,
    sys = mpc_sys,
    cost = tracking_cost,
    wref = sol['wsol'],
    tuning = tuning,
    lam_g_ref = lam_g0,
    sensitivities= sol['S'],
    options = opts
)

if ACADOS_SIM:

    # get system dae
    alg = user_input['dyn']

    # solver options
    opts = {}
    opts['integrator_type'] = 'IRK' # ERK, IRK, GNSF
    opts['nlp_solver_type'] = 'SQP' # SQP_RTI
    # opts['qp_solver_cond_N'] = Nmpc # ???
    opts['print_level'] = 1
    opts['sim_method_num_steps'] = 1
    opts['tf'] = Nmpc*user_input['ts']
    opts['nlp_solver_max_iter'] = 50
    opts['nlp_solver_step_length'] = 0.9
    opts['nlp_solver_tol_comp'] = 1e-5
    opts['nlp_solver_tol_eq'] = 1e-5
    opts['nlp_solver_tol_ineq'] = 1e-5
    opts['nlp_solver_tol_stat'] = 1e-5

    ctrls_acados = {}
    for ctrl_key in list(ctrls.keys()):
        if ctrl_key == 'EMPC':
            opts['hessian_approx'] = 'GAUSS_NEWTON'
            opts['qp_solver'] = 'PARTIAL_CONDENSING_HPIPM'
            opts['ext_cost_custom_hessian'] = False
            opts['custom_hessian'] = sol['S']['Hc']
        else:
            opts['hessian_approx'] = 'GAUSS_NEWTON'
            opts['qp_solver'] = 'PARTIAL_CONDENSING_HPIPM' # faster than full condensing
            opts['ext_cost_custom_hessian'] = False

        _, _ = ctrls[ctrl_key].generate(alg, opts = opts, name = 'awe_'+ctrl_key)
        ctrls_acados[ctrl_key+'_ACADOS'] = ctrls[ctrl_key]

# initialize and set-up open-loop simulation
alpha = np.linspace(-1.0, 1.0, alpha_steps+1) # deviation sweep grid
dz = 8 # max. deviation
x0 = sol['wsol']['x',0]
tgrid = [1/user_input['p']*i for i in range(Nmpc)]
tgridx = tgrid + [tgrid[-1]+1/user_input['p']]

# optimal stage cost and constraints for comparison
lOpt, hOpt = [], []
for k in range(Nmpc):
    lOpt.append(user_input['l'](sol['wsol']['x', k%user_input['p']], sol['wsol']['u',k%user_input['p']]).full()[0][0])
    hOpt.append(user_input['h'](sol['wsol']['x', k%user_input['p']], sol['wsol']['u',k%user_input['p']]).full())

# open loop simulation
import copy
log = []
log_acados = []
for alph in alpha:
    x_init = copy.deepcopy(x0)
    x_init[2] = x_init[2] + alph*dz
    x_init[0] = np.sqrt(-x_init[2]**2-x_init[1]**2+(l_t)**2)
    x_init[5] = -(x_init[0]*x_init[3] + x_init[1]*x_init[4]) / x_init[2]
    if TUNEMPC_SIM:
        log.append(clt.check_equivalence(ctrls, user_input['l'], user_input['h'], x0, x_init-x0, [1.0])[-1])
    if ACADOS_SIM:
        log_acados.append(clt.check_equivalence(
            ctrls_acados,
            user_input['l'],
            user_input['h'],
            x0,
            x_init-x0,
            [1.0],
            flag = 'acados')[-1])
    for name in list(ctrls.keys()):
        ctrls[name].reset()

# plotting options
alpha_plot = -1
lw = 2
ctrls_colors = {
    'EMPC': 'blue',
    'TUNEMPC': 'green',
    'TMPC_1': 'red',
    'TMPC_2': 'orange'
}
ctrls_lstyle = {
    'EMPC': 'solid',
    'TUNEMPC': 'dashed',
    'TMPC_1': 'dashdot',
    'TMPC_2': 'dotted'
}
ctrls_markers =  {
    'EMPC': '.',
    'TUNEMPC': 'o',
    'TMPC_1': '^',
    'TMPC_2': 'x'
}

if ACADOS_SIM:
    for ctrl_key in list(ctrls_acados.keys()):
        ctrls_colors[ctrl_key] = ctrls_colors[ctrl_key[:-7]]
        ctrls_lstyle[ctrl_key] = ctrls_lstyle[ctrl_key[:-7]]
        ctrls_markers[ctrl_key] = ctrls_markers[ctrl_key[:-7]]

ctrls_list = []
if TUNEMPC_SIM:
    ctrls_list += list(ctrls.keys())
if ACADOS_SIM:
    ctrls_list += list(ctrls_acados.keys())

# plot feedback equivalence
plt.figure(1)
for name in ctrls_list:
    if name not in ['EMPC', 'EMPC_ACADOS']:
        if name[-6:] == 'ACADOS':
            plot_log = log_acados
            EMPC_key = 'EMPC_ACADOS'
        else:
            plot_log = log
            EMPC_key = 'EMPC'
        feedback_norm = [
            np.linalg.norm(
                np.divide(
                    np.array(plot_log[k]['u'][name][0]) - np.array(plot_log[k]['u'][EMPC_key][0]),
                    np.array(plot_log[0]['u'][EMPC_key][0]))
            ) for k in range(len(alpha))]
        plt.plot(
            [dz*alph for alph in alpha],
            feedback_norm,
            marker = ctrls_markers[name],
            color = ctrls_colors[name],
            linestyle = ctrls_lstyle[name],
            markersize=2,
            linewidth=lw
                )
plt.grid(True)
plt.legend(ctrls_list[1:])
plt.title(r'$\Delta \pi_0^{\star}(\hat{x}_0) \ [-]$')
plt.xlabel(r'$\Delta z \ \mathrm{[m]}$')

# plot stage cost deviation over time
plt.figure(2)
for name in ctrls_list:
    if name[-6:] == 'ACADOS':
        plot_log = log_acados
    else:
        plot_log = log
    stage_cost_dev = [x[0] - x[1] for x in zip(plot_log[alpha_plot]['l'][name],lOpt)]
    plt.step(
        tgrid,
        stage_cost_dev,
        color = ctrls_colors[name],
        linestyle = ctrls_lstyle[name],
        linewidth=lw,
        where='post')
plt.legend(ctrls_list)
plt.grid(True)
plt.xlabel('t - [s]')
plt.title('Stage cost deviation')
plt.autoscale(enable=True, axis='x', tight=True)

# plot state deviation over time
plt.subplots(nx,1,sharex = True)
for i in range(nx):
    plt.subplot(nx,1,i+1)
    if i == 0:
        plt.title('State deviation')
    if i == nx:
        plt.xlabel('t - [s]')

    for name in ctrls_list:
        if name[-6:] == 'ACADOS':
            plot_log = log_acados
        else:
            plot_log = log
        plt.plot(
            tgridx,
            [plot_log[alpha_plot]['x'][name][j][i] - sol['wsol']['x',j][i] for j in range(Nmpc+1)],
            color = ctrls_colors[name],
            linestyle = ctrls_lstyle[name],
            linewidth=lw)
        plt.plot(tgridx, [0.0 for j in range(Nmpc+1)],  linestyle='--', color='black')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.grid(True)

# plot transient cost vs. alpha
plt.figure(4)
transient_cost = {}
for name in ctrls_list:
    if name[-6:] == 'ACADOS':
        plot_log = log_acados
    else:
        plot_log = log
    transient_cost[name] = []
    for i in range(len(alpha)):
            transient_cost[name].append(
                sum([x[0] - x[1] for x in zip(plot_log[i]['l'][name],lOpt)])
                )
    plt.plot(
        alpha,
        transient_cost[name],
        marker = ctrls_markers[name],
        color = ctrls_colors[name],
        linestyle = ctrls_lstyle[name],
        markersize = 2,
        linewidth=lw)
plt.grid(True)
plt.legend(ctrls_list)
plt.title('Transient cost')
plt.xlabel(r'$\alpha \ \mathrm{[-]}$')

if ACADOS_SIM:

    # plot time per iteration
    plt.figure(5)
    for name in ctrls_list:
        if name[-6:] == 'ACADOS':
            plot_log = log_acados
            plt.plot(
                alpha,
                [plot_log[k]['log'][name]['time_tot'][0]/plot_log[k]['log'][name]['sqp_iter'][0] for k in range(len(alpha))],
                marker = ctrls_markers[name],
                color = ctrls_colors[name[:-7]],
                linestyle = ctrls_lstyle[name],
                markersize = 2,
                linewidth = 2
            )
    plt.grid(True)
    plt.legend(list(plot_log[0]['log'].keys()))
    plt.title("Time per iteration")
    plt.xlabel('alpha [-]')
    plt.ylabel('t [s]')

    from statistics import mean
    mean_timings = {'time_lin': [], 'time_qp_xcond': [], 'time_qp': [],'time_tot': []}
    for name in ctrls_list:
        if name[-6:] == 'ACADOS':
            plot_log = log_acados
            for timing in list(mean_timings.keys()):
                mean_timings[timing].append(
                     mean([plot_log[k]['log'][name][timing][0][0]/plot_log[k]['log'][name]['sqp_iter'][0][0] for k in range(len(alpha))])
                )
    from tabulate import tabulate
    print(tabulate([
        ['time_tot']+mean_timings['time_tot'],
        ['time_lin']+mean_timings['time_lin'],
        ['time_qp']+mean_timings['time_qp'],
        ['time_qp_xcond']+mean_timings['time_qp_xcond']],
        headers=[' ']+list(ctrls.keys())))

plt.show()
