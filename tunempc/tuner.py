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
from tunempc.logger import Logger

class Tuner(object):

    def __init__(self, f, l, h = None, p = 1):

        """ Constructor 
        """

        # print license information
        self.__log_license_info()

        Logger.logger.info(60*'=')
        Logger.logger.info(18*' '+'Create Tuner instance...')
        Logger.logger.info(60*'=')
        Logger.logger.info('')

        # construct system
        sys = {'f': f}
        if h is not None:
            sys['h'] = h

        # detect nonlinear constraints
        Logger.logger.info('Detect nonlinear constraints...')

        self.__sys = preprocessing.input_formatting(sys)

        # problem dimensions
        self.__nx = sys['vars']['x'].shape[0]
        self.__nu = sys['vars']['u'].shape[0]
        self.__nw = self.__nx + self.__nu
        self.__p  = p
        if 'us' in sys['vars'].keys():
            self.__nus = sys['vars']['us'].shape[0]
        else:
            self.__nus = 0

        # construct p-periodic OCP
        if self.__p == 1:
            Logger.logger.info('Construct steady-state optimization problem...')
        else:
            Logger.logger.info('Construct {}-periodic optimal control problem...'.format(self.__p))


        self.__l = l
        self.__ocp = pocp.Pocp(sys = sys, cost = l, period = p)

        Logger.logger.info('')
        Logger.logger.info('Tuner instance created:')
        self.__log_problem_dimensions()

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
        Logger.logger.info(60*'=')
        Logger.logger.info(15*' '+'Solve optimization problem...')
        Logger.logger.info(60*'=')
        Logger.logger.info('')

        self.__w_sol = self.__ocp.solve(w0=w_init)
        self.__S     = self.__ocp.get_sensitivities()

        Logger.logger.info('')
        Logger.logger.info('Optimization problem solved.')
        Logger.logger.info('')

        return self.__w_sol

    def convexify(self, rho = 1.0, force = False, solver = 'cvxopt'):

        """ Compute positive definite stage cost matrices for a tracking NMPC
            scheme that is locally first-order equivalent to economic MPC.
        """


        # extract POCP sensitivities at optimal solution
        S = self.__S

        # extract Q, R, N
        Q = [S['H'][i][:self.__nx,:self.__nx] for i in range(self.__p)]
        R = [S['H'][i][self.__nx:,self.__nx:] for i in range(self.__p)]
        N = [S['H'][i][:self.__nx, self.__nx:] for i in range(self.__p)]

        # convexifier options
        opts = {'rho': rho, 'solver': solver, 'force': force}

        Logger.logger.info(60*'=')
        Logger.logger.info(15*' '+'Convexify Lagrangian Hessians...')
        Logger.logger.info(60*'=')
        Logger.logger.info('')

        dHc, _, _, _ = convexifier.convexify(S['A'], S['B'], Q, R, N, C = S['C_As'], G = S['G'], opts=opts)
        S['Hc'] = [S['H'][i] + dHc[i] for i in range(self.__p)] # build tuned MPC hessian

        return S['Hc']

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

    def __log_license_info(self):

        """ Print tunempc license info
        """

        license_info = []
        license_info += [80*'+']
        license_info += ['This is TuneMPC, a tool for economic tuning of tracking (N)MPC problems.']
        license_info += ['TuneMPC is free software; you can redistribute it and/or modify it under the terms']
        license_info += ['of the GNU Lesser General Public License as published by the Free Software Foundation']
        license_info += ['license. More information can be found at http://github.com/jdeschut/tunempc.']
        license_info += [80*'+']

        Logger.logger.info('')
        for line in license_info:
            Logger.logger.info(line)
        Logger.logger.info('')

        return None

    def __log_problem_dimensions(self):

        """ Logging of problem dimensions
        """

        Logger.logger.info('')
        Logger.logger.info('Number of states:................:  {}'.format(self.__nx))
        Logger.logger.info('Number of controls:..............:  {}'.format(self.__nu))
        if 'h' in self.__sys:
            Logger.logger.info('Number of linear constraints:....:  {}'.format(self.__sys['h'].size1_out(0)-self.__nus))
            Logger.logger.info('Number of nonlinear constraints:.:  {}'.format(self.__nus))
        Logger.logger.info('Steady-state period:.............:  {}'.format(self.__p))
        Logger.logger.info('')

        return None

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

    @property
    def S(self):
        "Sensitivity information"
        return self.__S