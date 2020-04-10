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
"""
Periodic OCP routines

:author: Jochem De Schutter
"""

import casadi.tools as ct
import casadi as ca
import numpy as np
import itertools
import collections
import tunempc.sqp_method as sqp_method
from tunempc.logger import Logger

class Pocp(object):

    def __init__(self, sys, cost, period = 1):

        """ Constructor
        """

        # store system dynamics
        self.__F    = sys['f']

        # store linear inequality constraints
        if 'h' in sys:
            self.__h = sys['h']
        else:
            self.__h = None

        # store slacked nonlinear inequality constraints
        if 'g' in sys:
            self.__gnl = sys['g']
        else:
            self.__gnl = None

        # store system variables
        self.__vars = sys['vars']
        self.__nx   = sys['vars']['x'].shape[0]
        self.__nu   = sys['vars']['u'].shape[0]
        if 'us' in sys['vars']: # slacks
            self.__ns = sys['vars']['us'].shape[0]

        # story economic cost function
        self.__cost = cost

        # desired steady state periods
        self.__N = period

        # construct OCP with sensitivities
        self.__parallelization = 'openmp'
        self.__mu_tresh = 1e-15 # treshold for active constraint detection
        self.__reg_slack = 1e-4 # slack regularization
        self.__construct_solver()
        self.__construct_sensitivity_funcs()

    def __construct_solver(self):

        """ Construct periodic NLP and solver.
        """

        # system variables and dimensions
        x = self.__vars['x']
        u = self.__vars['u']

        variables_entry = (
                ct.entry('x', shape = (self.__nx,), repeat = self.__N),
                ct.entry('u', shape = (self.__nu,), repeat = self.__N)
            )
        
        if 'us' in self.__vars:
            variables_entry += (
                ct.entry('us', shape = (self.__ns,), repeat = self.__N),
            )

        # nlp variables + bounds
        w = ct.struct_symMX([variables_entry])

        self.__lbw = w(-np.inf)
        self.__ubw = w(np.inf)

        # prepare dynamics and path constraints entry
        constraints_entry = (ct.entry('dyn', shape = (self.__nx,), repeat = self.__N),)
        if self.__h is not None:
            constraints_entry += (ct.entry('h', shape = self.__h.size1_out(0), repeat = self.__N),)
        if self.__gnl is not None:
            constraints_entry += (ct.entry('g', shape = self.__gnl.size1_out(0), repeat = self.__N),)

        # create general constraints structure
        g_struct = ct.struct_symMX([
            constraints_entry,
        ])

        # create symbolic constraint expressions
        map_args = collections.OrderedDict()
        map_args['x0'] = ct.horzcat(*w['x'])
        map_args['p']  = ct.horzcat(*w['u'])

        # evaluate function dynamics
        F_constr = ct.horzsplit(self.__F.map(self.__N, self.__parallelization)(**map_args)['xf'])

        # generate constraints
        constr = collections.OrderedDict()
        constr['dyn'] = [a - b for a,b in zip(F_constr, w['x',1:]+[w['x',0]])]

        if 'us' in self.__vars:
            map_args['us'] = ct.horzcat(*w['us'])
        if self.__h is not None:
            constr['h'] = ct.horzsplit(self.__h.map(self.__N,self.__parallelization)(*map_args.values()))
        if self.__gnl is not None:
            constr['g'] = ct.horzsplit(self.__gnl.map(self.__N,self.__parallelization)(*map_args.values()))

        # interleaving of constraints
        repeated_constr = list(itertools.chain.from_iterable(zip(*constr.values())))

        # fill in constraint structure
        self.__g = g_struct(ca.vertcat(*repeated_constr))

        # constraint bounds
        self.__lbg = g_struct(np.zeros(self.__g.shape))
        self.__ubg = g_struct(np.zeros(self.__g.shape))

        if self.__h is not None:
            self.__ubg['h',:] = np.inf

        # nlp cost
        cost_map_fun = self.__cost.map(self.__N,self.__parallelization)
        f = ca.sum2(cost_map_fun(map_args['x0'], map_args['p']))

        # add phase fixing cost
        self.__construct_phase_fixing_cost()
        alpha = ca.MX.sym('alpha')
        x0star = ca.MX.sym('x0star', self.__nx, 1)
        f += self.__phase_fix_fun(alpha, x0star, w['x',0])

        # add slack regularization
        # if 'us' in self.__vars:
        #     f += self.__reg_slack*ct.mtimes(ct.vertcat(*w['us']).T,ct.vertcat(*w['us']))

        # NLP parameters
        p = ca.vertcat(alpha, x0star)
        self.__w = w
        self.__g_fun = ca.Function('g_fun',[w,p],[self.__g])

        # create IP-solver
        prob = {'f': f, 'g': self.__g, 'x': w, 'p': p}
        opts = {'ipopt':{'linear_solver':'ma57'},'expand':False}
        if Logger.logger.getEffectiveLevel() > 10:
            opts['ipopt']['print_level'] = 0
            opts['print_time'] = 0
            opts['ipopt']['sb'] = 'yes'

        self.__solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        # create SQP-solver
        prob['lbg'] = self.__lbg
        prob['ubg'] = self.__ubg
        self.__sqp_solver = sqp_method.Sqp(prob)


        return None

    def solve(self, w0 = None):

        """
        Solve periodic OCP
        """

        # initialize
        if w0 is None:
            w0 = self.__w(0.0)
        
        # no phase fix cost
        self.__alpha = 0.0
        self.__x0star = np.zeros((self.__nx,1))
        p = ca.vertcat(
                self.__alpha,
                self.__x0star
            )

        # solve OCP
        Logger.logger.info('IPOPT pre-solve...')
        self.__sol = self.__solver(
            x0  = w0, 
            lbx = self.__lbw, 
            ubx = self.__ubw, 
            lbg = self.__lbg, 
            ubg = self.__ubg,
            p   = p
           )

        # fix phase
        if self.__N > 1:
            # prepare
            wsol = self.__w(self.__sol['x'])
            self.__alpha = 0.1
            self.__x0star = wsol['x',0]
            p = ca.vertcat(
                    self.__alpha,
                    self.__x0star
                )
            # solve
            Logger.logger.info('IPOPT pre-solve with phase-fix...')
            self.__sol = self.__solver(
                x0  = self.__sol['x'],
                lbx = self.__lbw,
                ubx = self.__ubw,
                lbg = self.__lbg,
                ubg = self.__ubg,
                p   = p
           )

        # solve with SQP (with active set QP solver) to retrieve active set
        Logger.logger.info('Solve with active-set based SQP method...')
        self.__sol = self.__sqp_solver.solve(self.__sol['x'], p, self.__sol['lam_g'])

        return self.__w(self.__sol['x'])

    def get_sensitivities(self):

        """
        Extract NLP sensitivities evaluated at the solution.
        """

        # solution
        wsol = self.__w(self.__sol['x'])

        # extract multipliers
        lam_g = self.__g(self.__sol['lam_g'])
        self.__lam_g = lam_g

        map_args = collections.OrderedDict()
        map_args['x'] = ct.horzcat(*wsol['x'])
        map_args['u'] = ct.horzcat(*wsol['u'])
        map_args['lam_g'] = ct.horzcat(*lam_g['dyn'])

        if 'us' in self.__vars:
            map_args['us'] = ct.horzcat(*wsol['us'])
            map_args['lam_s'] = ct.horzcat(*lam_g['g'])

        # sensitivity dict
        S = {}

        # dynamics sensitivities
        S['A']    = np.split(self.__jac_Fx(
            map_args['x'],
            map_args['u']
        ).full(), self.__N, axis = 1)

        S['B']    = np.split(self.__jac_Fu(
            map_args['x'],
            map_args['u']
        ).full(), self.__N, axis = 1)

        # extract active constraints
        if self.__h is not None:

            # add slacks to function args if necessary
            if 'us' in self.__vars:
                args = [map_args['x'],map_args['u'],map_args['us']]
            else:
                args = [map_args['x'],map_args['u']]

            # compute gradient of constraints
            mu_s  = lam_g['h']
            S['C']  = np.split(self.__jac_h(*args).full(), self.__N, axis = 1)
            S['e'] =  np.split(self.__h(*args).full(), self.__N, axis = 1)
            if 'g' in lam_g.keys():
                lam_s = lam_g['g']
                S['G']  = np.split(self.__jac_g(*args).full(), self.__N, axis = 1)
                S['r'] =  np.split(self.__gnl(*args).full(), self.__N, axis = 1)
            else:
                S['G']  = None
                S['r'] =  None

            # retrieve active set
            C_As = []
            self.__indeces_As = []
            for k in range(self.__N):
                C_active  = []
                index_active = []
                for i in range(mu_s[k].shape[0]):
                    if np.abs(mu_s[k][i].full()) > self.__mu_tresh:
                        C_active.append(S['C'][k][i,:])
                        index_active.append(i)
                if len(C_active) > 0:
                    C_As.append(ca.horzcat(*C_active).full().T)
                else:
                    C_As.append(None)
                self.__indeces_As.append(index_active)
            S['C_As'] = C_As

        else:
            S['C']    = None
            S['C_As'] = None
            S['G']    = None
            S['e']    = None
            S['r']    = None
            self.__indeces_As = None

        # compute hessian of lagrangian
        H = self.__sol['S']['H']
        M = self.__nx + self.__nu
        if 'us' in self.__vars:
            M += self.__ns
        S['H'] = [H[i*M:(i+1)*M,i*M:(i+1)*M] for i in range(self.__N)]

        # add slack regularization
        # if 'us' in self.__vars:
        #     for k in range(self.__N):
        #         Hlag[k][-self.__ns:,-self.__ns:] += self.__reg_slack*np.eye(self.__ns)
        # Hlag = np.split(self.__hess_lag(*map_args.values()).full(), self.__N, axis = 1)

        # # add phase fix contribution
        # if self.__N > 1:
        #     Hlag[0] += self.__hess_phase_fix(self.__alpha, self.__x0star, wsol['x',0])

        # compute cost function gradient
        if self.__h is not None:
            S['q'] = [- ca.mtimes(lam_g['h',i].T,S['C'][i]) for i in range(self.__N)]
        else:
            S['q'] = [np.zeros((1,self.__nx+self.__nu)) for i in range(self.__N)]

        return S

    def __construct_sensitivity_funcs(self):

        """ Construct functions for NLP sensitivity evaluations
        """

        # system variables
        x, u = self.__vars['x'], self.__vars['u']
        nx   = x.shape[0]
        wk   = ca.vertcat(x,u)
        if 'us' in self.__vars:
            us   = self.__vars['us']
            uhat = ca.vertcat(u,  us)
            wk   = ca.vertcat(wk, us)
        else:
            uhat = u

        # dynamics sensitivities
        x_next = self.__F(x0 = x, p = u)['xf'] # symbolic integrator evaluation

        self.__jac_Fx = ca.Function('jac_Fx',[x,u],[ca.jacobian(x_next,x)]).map(self.__N)
        self.__jac_Fu = ca.Function('jac_Fu',[x,u],[ca.jacobian(x_next,uhat)]).map(self.__N)

        self.__hess_F = []
        for i in range(nx):
            self.__hess_F.append(
                ca.Function('hess_F_'+str(i),[x,u],[ca.hessian(x_next[i],wk)[0]])
                )

        # cost sensitivities
        obj = self.__cost(x,u) # symbolic cost evaluation

        self.__jac_cost  = ca.Function('q',[x,u],[ca.jacobian(obj, wk)]).map(self.__N)
        hess_cost = ca.Function('hess_cost',[x,u],[ca.hessian(obj, wk)[0]])
        self.__hess_cost = hess_cost.map(self.__N)
        
        # phase fixing sensitivities
        alpha  = ca.MX.sym('alpha')
        x0star = ca.MX.sym('x0star', self.__nx, 1)
        phase_fix = self.__phase_fix_fun(alpha, x0star, x)
        self.__jac_phase_fix = ca.Function('jac_phase_fix',[alpha, x0star, x], [ca.jacobian(phase_fix,wk)])
        self.__hess_phase_fix = ca.Function('hess_phase_fix',[alpha, x0star, x], [ca.hessian(phase_fix,wk)[0]])

        # constraints sensitivities
        if self.__h is not None:
            if 'us' in self.__vars:
                constr = self.__h(x,u,us) # symbolic constraint evaluation
                self.__jac_h = ca.Function('jac_h',[x,u,us], [ca.jacobian(constr,wk)]).map(self.__N)

            else:
                constr = self.__h(x,u)
                self.__jac_h = ca.Function('jac_h',[x,u], [ca.jacobian(constr,wk)]).map(self.__N)

        if self.__gnl is not None:
            constr = self.__gnl(x,u,us)
            self.__jac_g = ca.Function('jac_g',[x,u,us], [ca.jacobian(constr,wk)]).map(self.__N)
            self.__hess_g = []
            for i in range(self.__ns):
                self.__hess_g.append(
                    ca.Function('hess_g_'+str(i),[x,u,us],[ca.hessian(constr[i],wk)[0]])
                )

        # hessian of the lagrangian
        hess_lag = hess_cost(x,u)
        lam_g = ct.MX.sym('lam_g',nx)
        hess_lag += sum([lam_g[i]*self.__hess_F[i](x,u) for i in range(nx)])
        lag_args = [x,u,lam_g]

        if 'us' in self.__vars:
            lam_s = ct.MX.sym('lam_s',self.__ns)
            hess_lag += sum([lam_s[i]*self.__hess_g[i](x,u,us) for i in range(self.__ns)])
            lag_args += [us,lam_s]
            hess_lag += ca.hessian(0.5*self.__reg_slack*ct.mtimes(us.T,us),wk)[0]

        self.__hess_lag = ca.Function('hess_lag_fun', [*lag_args], [hess_lag]).map(self.__N)

        return None

    def __construct_phase_fixing_cost(self):

        """
        Construct phase fixing cost function part.
        Only applied to initial state.
        """

        alpha  = ca.MX.sym('alpha')
        x0star = ca.MX.sym('x0star', self.__nx, 1)
        x0     = ca.MX.sym('x0star', self.__nx, 1)
        phase_fix = 0.5*alpha * ca.mtimes((x0 - x0star).T,x0 - x0star)
        self.__phase_fix_fun = ca.Function('phase_fix', [alpha, x0star, x0], [phase_fix])

        return None

    @property
    def w(self):
        "OCP variables structure"
        return self.__w

    @property
    def indeces_As(self):
        "Active constraints indeces along trajectory"
        return self.__indeces_As

    @property
    def lam_g(self):
        "Constraints lagrange multipliers"
        return self.__lam_g

    @property
    def g(self):
        "Constraints"
        return self.__g

    @property
    def g_fun(self):
        "Constraints function"
        return self.__g_fun