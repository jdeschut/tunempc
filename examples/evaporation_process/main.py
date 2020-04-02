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
"""Evaporation process example as described in:

A tracking MPC formulation that is locally equivalent to economic MPC 
M. Zanon, S. Gros, M. Diehl 
Journal of Process Control 2016
(section 8) 

:author: Jochem De Schutter

"""

import tunempc
import tunempc.pmpc
import tunempc.closed_loop_tools as clt
import numpy as np
import casadi as ca
import casadi.tools as ct
import matplotlib.pyplot as plt

def problem_data():

    """ Problem data, numeric constants,...
    """

    data = {}
    data['a'] = 0.5616
    data['b'] = 0.3126
    data['c'] = 48.43
    data['d'] = 0.507
    data['e'] = 55.0
    data['f'] = 0.1538
    data['g'] = 90.0
    data['h'] = 0.16

    data['M'] = 20.0
    data['C'] = 4.0
    data['UA2'] = 6.84
    data['Cp'] = 0.07
    data['lam'] = 38.5
    data['lams'] = 36.6
    data['F1'] = 10.0
    data ['X1'] = 5.0
    data['F3'] = 50.0
    data['T1'] = 40.0
    data['T200'] = 25.0

    return data

def intermediate_vars(x, u, data):

    """ Intermediate model variables
    """

    data['T2'] = data['a']*x['P2'] + data['b']*x['X2'] + data['c']
    data['T3'] = data['d']*x['P2'] + data['e']
    data['T100'] = data['f']*u['P100'] + data['g']
    data['UA1'] = data['h']*(data['F1']+data['F3'])
    data['Q100'] = data['UA1']*(data['T100'] - data['T2'])
    data['F100'] = data['Q100']/data['lams']
    data['Q200'] = data['UA2']*(data['T3']-data['T200'])/(1.0 + data['UA2']/(2.0*data['Cp']*u['F200']))
    data['F5'] = data['Q200']/data['lam']
    data['F4'] = (data['Q100']-data['F1']*data['Cp']*(data['T2']-data['T1']))/data['lam']
    data['F2'] = data['F1'] - data['F4'] 

    return data

def dynamics(x, u, data):

    """ System dynamics function (discrete time)
    """

    # state derivative expression
    xdot = ca.vertcat(
        (data['F1']*data['X1'] - data['F2']*x['X2'])/data['M'],
        (data['F4'] - data['F5'])/data['C']
        )

    # create ode for integrator
    ode = {'x':x, 'p':u,'ode': xdot}

    return [ca.integrator('F','collocation',ode,{'tf':1}), ode]

def vars():

    """ System states and controls
    """

    x = ct.struct_symMX(['X2','P2'])
    u = ct.struct_symMX(['P100','F200'])

    return x, u

def objective(x, u, data):

    """ Economic objective function
    """
    
    # cost definition
    obj = 10.09*(data['F2']+data['F3']) + 600.0*data['F100'] + 0.6*u['F200']

    return ca.Function('economic_cost',[x,u],[obj])

def constraints(x, u, data):
    
    """ Path inequality constraints function (convention h(x,u) >= 0)
    """

    constr = ca.vertcat(
        x['X2'] - 25.0,
        x['P2'] - 40.0,
        80.0 - x['P2'],
        400.0 - u['P100'],
        400.0 - u['F200'],
    )

    return ca.Function('h', [x,u], [constr])


# set-up system
x, u = vars()
data = intermediate_vars(x,u, problem_data())
nx = x.shape[0]
nu = u.shape[0]

tuner = tunempc.Tuner(
    f = dynamics(x, u, data)[0],
    l = objective(x,u,data),
    h = constraints(x,u, data),
    p = 1
)

# solve
w0 = ca.vertcat(*[25.0, 49.743, 191.713, 215.888])
wsol = tuner.solve_ocp(w0)
Hc = tuner.convexify(rho = 1e-3, force = False, solver='mosek')

# nmpc horizon length
N = 200

# gradient
S  = tuner.S

# economic mpc controller
ctrls = {}
sys  = tuner.sys
ctrls['economic'] = tuner.create_mpc('economic',N = N)

# normal tracking mpc controller
tuningTn = {'H': [np.diag([10.0, 10.0, 0.1, 0.1])], 'q': S['q']}
ctrls['tracking'] = tuner.create_mpc('tracking',N = N, tuning = tuningTn)

# tuned tracking mpc controller
ctrls['tuned'] = tuner.create_mpc('tuned',N = N)


# ======================
# OPTIONS
# ======================
COST_TYPE = 'linear_ls'
MPC_TYPE = 'tuned'
TERMINAL_CONSTR = True
INEQ_CONSTR = True
PLOT_PREDICTION = True
N = 200
Nsim = 30
dP2 = 0.1 # initial perturbation
# alpha = [1.0]
# log = clt.check_equivalence(ctrls, objective(x,u,data), sys['h'], wsol['x',0], ca.vertcat(0.0, dP2), alpha)

# ======================
# ACADOS MODEL
# ======================
import os
os.system('rm *.json')
os.system('rm -rf c_generated_code')
ode = ca.Function('ode',[x.cat,u.cat],[dynamics(x,u,data)[1]['ode']])
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
xref = np.squeeze(wsol['x',0].full())
uref = np.squeeze(wsol['u',0].full())
model = AcadosModel()
xdot = ca.MX.sym('xdot',tuner.sys['vars']['x'].shape[0])
model.xdot = xdot
model.f_impl_expr = xdot - ode(tuner.sys['vars']['x'], tuner.sys['vars']['u'])
model.f_expl_expr = xdot
model.x = tuner.sys['vars']['x']
model.u = tuner.sys['vars']['u']
model.p = []
# model.z = z
model.name = 'evaporation_process'

# if INEQ_CONSTR:
#     model.con_h_expr = tuner.sys['h'](tuner.sys['vars']['x'], tuner.sys['vars']['u'])

if COST_TYPE == 'external':
    if MPC_TYPE == 'economic':
        model.cost_expr_ext_cost = tuner.l(tuner.sys['vars']['x'], tuner.sys['vars']['u'])
    elif MPC_TYPE == 'tuned':
        w = ca.vertcat(tuner.sys['vars']['x'] - xref, tuner.sys['vars']['u']-uref)
        model.cost_expr_ext_cost = 0.5*ct.mtimes(ct.mtimes(w.T, tuner.S['Hc'][0].full()), w) + ct.mtimes(q[0],w)

# ======================
# ACADOS OCP
# ======================

ocp = AcadosOcp()
ocp.model = model
ny = nx + nu
ny_e = nx

# set dimensions
ocp.dims.N = N

# set cost module
if COST_TYPE == 'external':
    ocp.cost.cost_type = 'EXTERNAL'
elif COST_TYPE == 'linear_ls':
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.W = tuner.S['Hc'][0].full()
    ocp.cost.W_e = np.zeros((nx,nx))
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx:,:] = np.eye(nu)
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref  = np.squeeze(ca.vertcat(xref,uref).full())
    ocp.cost.yref_e = np.zeros((ny_e, ))

ocp.cost.cost_type_e = 'LINEAR_LS'

# initial condition
ocp.constraints.x0 = xref

# set inequality constraints
if INEQ_CONSTR:
    ocp.constraints.constr_type = 'BGH'
    C = S['C'][0][:,:nx]
    D = S['C'][0][:,nx:]
    lg = -S['e'][0] + ct.mtimes(C,xref).full() + ct.mtimes(D,uref).full()
    ocp.constraints.lg = np.squeeze(lg)
    ocp.constraints.ug = 1e15*np.ones((5,))
    ocp.constraints.C  = C
    ocp.constraints.D  = D

# terminal constraint
if TERMINAL_CONSTR:
    ocp.constraints.lbx_e = xref
    ocp.constraints.ubx_e = xref
    ocp.constraints.Jbx_e = np.eye(nx)

ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES' # PARTIAL_CONDENSING_HPIPM
if COST_TYPE == 'linear_ls':
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'IRK'
ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI
ocp.solver_options.qp_solver_cond_N = 2 # ???
ocp.solver_options.print_level = 1
ocp.solver_options.sim_method_num_steps = 20
# ocp.solver_options.nlp_solver_max_iter = 30
# ocp.solver_options.nlp_solver_step_length = 0.9
# set prediction horizon
ocp.solver_options.tf = N

acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')

simX = np.ndarray((Nsim+1, nx))
simU = np.ndarray((Nsim, nu))

xcurrent = xref

# initialize
for i in range(N):
    acados_ocp_solver.set(i, "x", xref)
    acados_ocp_solver.set(i, "u", uref)

# pre-solve for warm-starting:
acados_ocp_solver.set(0, "lbx", xcurrent)
acados_ocp_solver.set(0, "ubx", xcurrent)
status = acados_ocp_solver.solve()

# perturb inital state
xcurrent += np.array([0.0, dP2])
simX[0,:] = xcurrent

# closed loop
for i in range(Nsim):
    print(i)
    # solve ocp
    acados_ocp_solver.set(0, "lbx", xcurrent)
    acados_ocp_solver.set(0, "ubx", xcurrent)

    status = acados_ocp_solver.solve()

    if status != 0:
        raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

    simU[i,:] = acados_ocp_solver.get(0, "u")

    # simulate system
    acados_integrator.set("x", xcurrent)
    acados_integrator.set("u", simU[i,:])

    status = acados_integrator.solve()
    if status != 0:
        raise Exception('acados integrator returned status {}. Exiting.'.format(status))

    # update state
    xcurrent = acados_integrator.get("x")
    simX[i+1,:] = xcurrent

    # prediction
    if PLOT_PREDICTION:
        predX = np.ndarray((N+1, nx))
        predU = np.ndarray((N, nu))
        for k in range(N):
            predX[k,:] = acados_ocp_solver.get(k, "x")
            predU[k,:] = acados_ocp_solver.get(k, "u")
        predX[N,:] = acados_ocp_solver.get(N, "x")
        for k in range(nu):
            plt.subplot(nx+nu, 1, k+1)
            plt.step(range(N), predU[:,k], color='r', where='post')
            if k == 0:
                plt.title('closed-loop simulation')
            plt.hlines(400.0, 0, N-1, linestyles='dashed', alpha=0.7)
            plt.ylabel('$u$')
            plt.xlabel('$t$')
            plt.grid()

        for k in range(nx):
                plt.subplot(nx+nu, 1, k+nu+1)
                plt.plot(range(N+1), predX[:,k], label='true')
                if k == 0:
                    plt.hlines(25.0, 0, N, linestyles='dashed', alpha=0.7)
                if k == 1:
                    plt.hlines(40.0, 0, N, linestyles='dashed', alpha=0.7)
                    plt.hlines(80.0, 0, N, linestyles='dashed', alpha=0.7)
                plt.xlabel('$t$')
                plt.grid()
                plt.legend(loc=1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
        plt.show()

for i in range(nu):
    plt.subplot(nx+nu, 1, i+1)
    plt.step(range(Nsim), simU[:,i], color='r', where='post')
    if i == 0:
        plt.title('closed-loop simulation')
    plt.hlines(400.0, 0, Nsim-1, linestyles='dashed', alpha=0.7)
    plt.ylabel('$u$')
    plt.xlabel('$t$')
    plt.grid()

for i in range(nx):
    plt.subplot(nx+nu, 1, i+nu+1)
    plt.plot(range(Nsim+1), simX[:,i], label='true')
    if i == 0:
        plt.hlines(25.0, 0, Nsim, linestyles='dashed', alpha=0.7)
    if i == 1:
        plt.hlines(40.0, 0, Nsim, linestyles='dashed', alpha=0.7)
        plt.hlines(80.0, 0, Nsim, linestyles='dashed', alpha=0.7)
    plt.xlabel('$t$')
    plt.grid()
    plt.legend(loc=1)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
plt.show()


# # plot feedback controls to check equivalence
# for name in list(ctrls.keys()):
#     for i in range(nu):
#         plt.figure(i)
#         plt.plot(alpha, [log[j]['u'][name][0][i] for j in range(len(alpha))])
#         plt.legend(list(ctrls.keys()))