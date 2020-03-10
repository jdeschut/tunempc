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
"""
Tuner high level class

:author: Jochem De Schutter
"""

import tunempc.preprocessing as preprocessing
import tunempc.pocp as pocp
import tunempc.convexifier as convexifier
import tunempc.pmpc as pmpc
import casadi as ca
import casadi.tools as ct
import copy

class Tuner(object):

    def __init__(self, f, l, h = None, p = 1):

        """ Constructor 
        """

        # construct system
        sys = {'f': f}
        if h is not None:
            sys['h'] = h


        # detect nonlinear constraints
        self.__sys = preprocessing.input_formatting(sys)

        # system dimensions
        self.__nx = sys['vars']['x'].shape[0]
        self.__nu = sys['vars']['u'].shape[0]
        if 'us' in sys['vars'].keys():
            self.__nus = sys['vars']['us'].shape[0]
        else:
            self.__nus = 0
        self.__nw = self.__nx + self.__nu
        self.__p  = p

        # construct p-periodic OCP
        self.__l = l
        self.__ocp = pocp.Pocp(sys = sys, cost = l, period = p)

        return None
    
    def solve_ocp(self, w0 = None, lam0 = None):

        """ Solve periodic optimal control problem.
        """

        # check input dimensions
        w0_shape = self.__p*(self.__nx+self.__nu)
        assert w0.shape[0] == w0_shape, \
            'Incorrect dimensions of input variable w0: expected {}x1, \
                but reveived {}x{}'.format(w0_shape, w0.shape[0], w0.shape[1])

        # initialize
        w_init = self.__ocp.w(0.0)
        if w0 is not None:

            for k in range(self.__p):

                # initialize primal variables
                w_init['x',k] = w0[k*self.__nw: k*self.__nw+self.__nx]
                w_init['u',k] = w0[k*self.__nw+self.__nx:(k+1)*self.__nw]

                # initialize slacks
                if 'g' in self.__sys:
                    w_init['us',k] = self.__sys['g'](w_init['x',k], w_init['u',k],0.0)

        # solve OCP
        self.__w_sol = self.__ocp.solve(w0=w_init)

        return self.__w_sol

    def convexify(self, rho = 1.0, force = False, solver = 'cvxopt'):

        """ Compute positive definite stage cost matrices for a tracking NMPC
            scheme that is locally first-order equivalent to economic MPC.
        """


        # extract POCP sensitivities at optimal solution
        [H, q, A, B, D] = self.__ocp.get_sensitivities()

        # extract Q, R, N
        Q = [H[i][:self.__nx,:self.__nx] for i in range(self.__p)]
        R = [H[i][self.__nx:,self.__nx:] for i in range(self.__p)]
        N = [H[i][:self.__nx, self.__nx:] for i in range(self.__p)]

        opts = {'rho': rho, 'solver': solver, 'force': force}
        dHc, _, _, _ = convexifier.convexify(A, B, Q, R, N, D, opts)
        Hc = [H[i] + dHc[i] for i in range(self.__p)] # build tuned MPC hessian

        self.__S = {'H': H,'q': q,'A': A,'B': B,'D': D,'Hc': Hc}

        return Hc

    def create_mpc(self, mpc_type, N, opts = {}, tuning = None):

        """ Create MPC controller of user-defined type  and horizon
        for given system dynamics, constraints and cost function.
        """

        if mpc_type not in ['economic','tuned','tracking']:
            raise ValueError('Provided MPC type not supported.')
        
        if 'slack_flag' in opts.keys():
            mpc_sys = preprocessing.add_mpc_slacks(
                copy.deepcopy(self.__sys),
                self.__ocp.lam_g,
                self.__ocp.indeces_As,
                slack_flag = opts['slack_flag']
            )
        else:
            mpc_sys = self.__sys

        if mpc_type == 'economic':
            mpc = pmpc.Pmpc(N = N, sys = mpc_sys, cost = self.__l, wref = self.__w_sol, lam_g_ref=self.__ocp.lam_g, options=opts)
        else:
            cost = self.__tracking_cost(self.__nx+self.__nu+self.__nus)
            lam_g0 = copy.deepcopy(self.__ocp.lam_g)
            lam_g0['dyn'] = 0.0
            if 'g' in lam_g0.keys():
                lam_g0['g'] = 0.0

            if mpc_type == 'tracking':
                if tuning is None:
                    raise ValueError('Tracking type MPC controller requires user-provided tuning!')
            elif mpc_type == 'tuned':
                tuning = {'H': self.__S['Hc'], 'q': self.__S['q']}
            mpc = pmpc.Pmpc(N = N, sys = mpc_sys, cost = cost, wref = self.__w_sol, tuning = tuning, lam_g_ref=lam_g0, options=opts)
        
        return mpc

    def __tracking_cost(self, nw):

        """ Create tracking cost function
        """

        # reference parameters
        w =  ca.MX.sym('w', (nw, 1))
        wref = ca.MX.sym('wref', (nw, 1))
        H = ca.MX.sym('H', (nw, nw))
        q = ca.MX.sym('H', (nw, 1))

        # cost definition
        dw   = w - wref
        obj = 0.5*ct.mtimes(dw.T, ct.mtimes(H, dw)) + ct.mtimes(q.T,dw)

        return ca.Function('tracking_cost',[w, wref, H, q],[obj])

    @property
    def pocp(self):
        "Periodic optimal control problem"
        return self.__ocp

    @property
    def sys(self):
        "Reformulated system dynamics"
        return self.__sys

    @property
    def p(self):
        "Period"
        return self.__p

    @property
    def w_sol(self):
        "Optimal periodic orbit"
        return self.__w_sol

    @property
    def l(self):
        "Economic cost function"
        return self.__l