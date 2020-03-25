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
Periodic MPC routines

:author: Jochem De Schutter
"""

import casadi.tools as ct
import casadi as ca
import numpy as np
import itertools
import collections
import tunempc.sqp_method as sqp_method
from tunempc.logger import Logger

class Pmpc(object):

    def __init__(self, N, sys, cost, wref = None, tuning = None, lam_g_ref = None, options = {}):
        
        """ Constructor
        """
        
        # store construction data
        self.__N    = N
        self.__vars = sys['vars']
        self.__nx   = sys['vars']['x'].shape[0]
        self.__nu   = sys['vars']['u'].shape[0]
        
        # nonlinear inequalities slacks
        if 'us' in sys['vars']:
            self.__ns = sys['vars']['us'].shape[0]
        else:
            self.__ns = 0

        # mpc slacks
        if 'usc' in sys['vars']:
            self.__nsc = sys['vars']['usc'].shape[0]
            self.__scost = sys['scost']
        else:
            self.__nsc = 0

        # store system dynamics
        self.__F    = sys['f']

        # store path constraints
        if 'h' in sys:
            self.__h = sys['h']
        else:
            self.__h = None

        # store slacked nonlinear inequality constraints
        if 'g' in sys:
            self.__gnl = sys['g']
        else:
            self.__gnl = None

        self.__cost = cost

        # set options
        self.__options = self.__default_options()
        for option in options:
            if option in self.__options:
                self.__options[option] = options[option]
            else:
                raise ValueError('Unknown option for Pmpc class instance: "{}"'.format(option))

        # detect cost-type
        if self.__cost.n_in() == 2:

            # cost function of the form: l(x,u)
            self.__type = 'economic'
            
            # no tuning required
            tuning = None 

            if self.__options['hessian_approximation'] == 'gauss_newton':
                self.__options['hessian_approximation'] = 'exact'
                Logger.logger.warning('Gauss-Newton Hessian approximation cannot be applied for economic MPC problem. Switched to exact Hessian.')

        else:

            # cost function of the form: (w-wref)'*H*(w-wref) + q'w
            self.__type = 'tracking'

            # check if tuning matrices are provided
            assert tuning != None, 'Provide tuning matrices for tracking MPC!'

        # periodicity operator
        self.__p_operator = self.__options['p_operator']

        # construct MPC solver
        self.__construct_solver()

        # periodic indexing
        self.__index = 0

        # create periodic reference
        assert wref   != None, 'Provide reference trajectory!'
        self.__create_reference(wref, tuning, lam_g_ref)

        # initialize log
        self.__initialize_log()

        # solver initial guess
        self.__set_initial_guess()

        return None

    def __default_options(self):

        # default options
        opts = {
            'hessian_approximation': 'exact',
            'ipopt_presolve': False,
            'max_iter': 2000,
            'p_operator': None,
            'slack_flag': 'none'
        }

        return opts

    def __construct_solver(self):

        """ Construct periodic MPC solver
        """

        # system variables and dimensions
        x = self.__vars['x']
        u = self.__vars['u']

        # NLP parameters

        if self.__type == 'economic':

            # parameters
            self.__p = ct.struct_symMX([
                ct.entry('x0', shape = (self.__nx,1)),
                ct.entry('xN', shape = (self.__nx,1))
            ])

            # reassign for brevity
            x0 = self.__p['x0']
            xN = self.__p['xN']

        if self.__type == 'tracking':
            ref_vars = (
                    ct.entry('x', shape = (self.__nx,), repeat = self.__N+1),
                    ct.entry('u', shape = (self.__nu,), repeat = self.__N)
                )
            
            if 'us' in self.__vars:
                ref_vars += (ct.entry('us', shape = (self.__ns,), repeat = self.__N),)

            # reference trajectory
            wref = ct.struct_symMX([ref_vars])

            nw = self.__nx + self.__nu + self.__ns
            tuning = ct.struct_symMX([ # tracking tuning
                ct.entry('H', shape = (nw, nw), repeat = self.__N),
                ct.entry('q', shape = (nw, 1),  repeat = self.__N)
            ])

            # parameters
            self.__p = ct.struct_symMX([
                ct.entry('x0',     shape = (self.__nx,)),
                ct.entry('wref',   struct = wref),
                ct.entry('tuning', struct = tuning)
            ])

            # reassign for brevity
            x0     = self.__p['x0']
            wref   = self.__p.prefix['wref']
            tuning = self.__p.prefix['tuning']
            xN     = wref['x',-1]

        # NLP variables
        variables_entry = (
                ct.entry('x', shape = (self.__nx,), repeat = self.__N+1),
                ct.entry('u', shape = (self.__nu,), repeat = self.__N)
            )
        
        if 'us' in self.__vars:
            variables_entry += (
                ct.entry('us', shape = (self.__ns,), repeat = self.__N),
            )

        self.__wref = ct.struct_symMX([variables_entry]) # structure of reference

        if 'usc' in self.__vars:
            variables_entry += (
                ct.entry('usc', shape = (self.__nsc,), repeat = self.__N),
            )

        # nlp variables + bounds
        w = ct.struct_symMX([variables_entry])

        # variable bounds are implemented as inequalities
        self.__lbw = w(-np.inf)
        self.__ubw = w(np.inf)

        # prepare dynamics and path constraints entry
        constraints_entry = (ct.entry('dyn', shape = (self.__nx,), repeat = self.__N),)
        if self.__gnl is not None:
            constraints_entry += (ct.entry('g', shape = self.__gnl.size1_out(0), repeat = self.__N),)
        if self.__h is not None:
            constraints_entry += (ct.entry('h', shape = self.__h.size1_out(0), repeat = self.__N),)

        # terminal constraint
        if self.__p_operator is not None:
            nx_term = self.__p_operator.size1_out(0)
        else:
            nx_term = self.__nx

        # create general constraints structure
        g_struct = ct.struct_symMX([
            ct.entry('init', shape = (self.__nx,1)),
            constraints_entry,
            ct.entry('term', shape = (nx_term,1))
        ])

        # create symbolic constraint expressions
        map_args = collections.OrderedDict()
        map_args['x0'] = ct.horzcat(*w['x',:-1])
        map_args['p']  = ct.horzcat(*w['u'])
        F_constr = ct.horzsplit(self.__F.map(self.__N)(**map_args)['xf'])

        # generate constraints
        constr = collections.OrderedDict()
        constr['dyn'] = [a - b for a,b in zip(F_constr, w['x',1:])]
        if 'us' in self.__vars:
            map_args['us'] = ct.horzcat(*w['us'])

        if self.__gnl is not None:
            constr['g'] = ct.horzsplit(self.__gnl.map(self.__N)(*map_args.values()))

        if 'usc' in self.__vars:
            map_args['usc'] = ct.horzcat(*w['usc'])

        if self.__h is not None:
            constr['h'] = ct.horzsplit(self.__h.map(self.__N)(*map_args.values()))

        repeated_constr = list(itertools.chain.from_iterable(zip(*constr.values())))

        if self.__p_operator is not None:
            term_constraint = self.__p_operator(w['x',-1] - xN)
        else:
            term_constraint = w['x',-1] - xN

        self.__g = g_struct(ca.vertcat(
            w['x',0] - x0,
            *repeated_constr,
            term_constraint
        ))

        self.__lbg = g_struct(np.zeros(self.__g.shape))
        self.__ubg = g_struct(np.zeros(self.__g.shape))
        if self.__h is not None:
            self.__ubg['h',:] = np.inf

        # nlp cost
        cost_map = self.__cost.map(self.__N)

        if self.__type == 'economic':

            cost_args = [ct.horzcat(*w['x',:-1]), ct.horzcat(*w['u'])]

        elif self.__type == 'tracking':

            if self.__ns != 0:
                cost_args_w = ct.horzcat(
                    *[ct.vertcat(w['x',k], w['u',k],w['us',k]) for k in range(self.__N)]
                    )
                cost_args_w_ref = ct.horzcat(
                    *[ct.vertcat(wref['x',k], wref['u',k], wref['us',k]) for k in range(self.__N)]
                    )
            else:
                cost_args_w = ct.horzcat(
                    *[ct.vertcat(w['x',k], w['u',k]) for k in range(self.__N)]
                    )
                cost_args_w_ref = ct.horzcat(
                    *[ct.vertcat(wref['x',k], wref['u',k]) for k in range(self.__N)]
                    )

            cost_args = [
                cost_args_w,
                cost_args_w_ref,
                ct.horzcat(*tuning['H']),
                ct.horzcat(*tuning['q'])
            ]

            if self.__options['hessian_approximation'] == 'gauss_newton':

                if 'usc' not in self.__vars:
                    hess_gn = ct.diagcat(*tuning['H'], ca.DM.zeros(self.__nx, self.__nx))
                else:
                    hess_block = list(itertools.chain.from_iterable(zip(tuning['H'],[ca.DM.zeros(self.__nsc, self.__nsc)]*self.__N)))
                    hess_gn = ct.diagcat(*hess_block, ca.DM.zeros(self.__nx, self.__nx))

        J = ca.sum2(cost_map(*cost_args))

        # add cost on slacks
        if 'usc' in self.__vars:
            J += ca.sum2(ct.mtimes(self.__scost.T,ct.horzcat(*w['usc'])))

        # create solver
        prob = {'f': J, 'g': self.__g, 'x': w, 'p': self.__p}
        self.__w = w
        self.__g_fun = ca.Function('g_fun',[self.__w,self.__p],[self.__g])

        # create IPOPT-solver instance if needed
        if self.__options['ipopt_presolve']:
            opts = {'ipopt':{'linear_solver':'ma57','print_level':0},'expand':False}
            self.__solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        # create hessian approximation function
        if self.__options['hessian_approximation'] == 'gauss_newton':
            lam_g = ca.MX.sym('lam_g',self.__g.shape) # will not be used
            hess_approx = ca.Function('hess_approx', [self.__w, self.__p, lam_g], [hess_gn])
        elif self.__options['hessian_approximation'] == 'exact':
            hess_approx = 'exact'

        # create sqp solver
        prob['lbg'] = self.__lbg
        prob['ubg'] = self.__ubg
        sqp_opts = {
            'hessian_approximation': hess_approx,
            'max_iter': self.__options['max_iter']
        }
        self.__sqp_solver = sqp_method.Sqp(prob, sqp_opts)

    def step(self, x0):

        """ Compute periodic MPC feedback control for given initial condition.
        """

        # reset periodic indexing if necessary
        self.__index = self.__index%len(self.__ref)

        # update nlp parameters
        p0 = self.__p(0.0)
        p0['x0'] = x0

        if self.__type == 'economic':

            p0['xN'] = self.__ref[self.__index][-x0.shape[0]:]

        elif self.__type == 'tracking':

            p0['wref'] = self.__ref[self.__index]
            p0['tuning','H'] = self.__Href[self.__index]
            p0['tuning','q'] = self.__qref[self.__index]

        # pre-solve NLP with IPOPT for globalization
        if self.__options['ipopt_presolve']:

            ipopt_sol = self.__solver(
                x0  = self.__w0,
                lbg = self.__lbg,
                ubg = self.__ubg,
                p   = p0
            )

            self.__w0 = self.__w(ipopt_sol['x'])
            self.__lam_g0 = self.__g(ipopt_sol['lam_g'])

        # solve NLP
        sol = self.__sqp_solver.solve(self.__w0.cat, p0.cat, self.__lam_g0.cat)

        # store solution
        self.__g_sol = self.__g(self.__g_fun(sol['x'], p0))
        self.__w_sol = self.__w(sol['x'])
        self.__extract_solver_stats()

        # shift reference
        self.__index += 1

        # update initial guess
        self.__w0, self.__lam_g0 = self.__shift_initial_guess(
            self.__w_sol,
            self.__g(sol['lam_g'])
            )

        return self.__w_sol['u',0]

    def __create_reference(self, wref, tuning, lam_g_ref):

        """ Create periodic reference and tuning data.
        """

        # period of reference
        Nref = len(wref['u'])

        # create reference and tuning sequence
        # for each starting point in period
        ref_pr = []
        ref_du = []
        H   = []
        q   = []

        for k in range(Nref):
            
            # reference primal solution
            refk = []
            for j in range(self.__N):

                refk += [
                    wref['x', (k+j)%Nref],
                    wref['u', (k+j)%Nref]
                ]

                if 'us' in self.__vars:
                    refk += [wref['us', (k+j)%Nref]]
            
            refk.append(wref['x', (k+self.__N)%Nref])

            # reference dual solution
            lamgk = self.__g(0.0)
            lamgk['init'] = -lam_g_ref['dyn',(k-1)%Nref]
            for j in range(self.__N):
                lamgk['dyn',j] = lam_g_ref['dyn',(k+j)%Nref]
                if 'g' in list(lamgk.keys()):
                    lamgk['g',j] = lam_g_ref['g', (k+j)%Nref]
                if 'h' in list(lamgk.keys()):
                    lam_h = [lam_g_ref['h', (k+j)%Nref]]
                    if 'usc' in self.__vars:
                        lam_h += [-self.__scost] # TODO not entirely correct

                    lamgk['h',j] = ct.vertcat(*lam_h)
            if self.__p_operator is not None:
                lamgk['term'] = self.__p_operator(lam_g_ref['dyn',(k+self.__N-1)%Nref])
            else:
                lamgk['term'] = lam_g_ref['dyn',(k+self.__N-1)%Nref]

            ref_pr.append(ct.vertcat(*refk))
            ref_du.append(lamgk.cat)

            if tuning is not None:
                H.append([tuning['H'][(k+j)%Nref] for j in range(self.__N)])
                q.append([tuning['q'][(k+j)%Nref] for j in range(self.__N)])

        self.__ref    = ref_pr
        self.__ref_du = ref_du
        self.__Href = H
        self.__qref = q

        return None

    def __initialize_log(self):

        self.__log = {
            'cpu': [],
            'iter': [],
            'f': [],
            'status': [],
            'sol_x': [],
            'lam_x': [],
            'lam_g': [],
            'u0': [],
            'nACtot': [],
            'nAC': [],
            'idx_AC': [],
            'nAS': []
        }

        return None

    def __extract_solver_stats(self):

        info = self.__sqp_solver.stats
        self.__log['cpu'].append(info['t_wall_total'])
        self.__log['iter'].append(info['iter_count'])
        self.__log['status'].append(info['return_status'])
        self.__log['sol_x'].append(info['x'])
        self.__log['lam_g'].append(info['lam_g'])
        self.__log['f'].append(info['f'])
        self.__log['u0'].append(self.__w(info['x'])['u',0])
        self.__log['nACtot'].append(info['nAC'])
        nAC, idx_AC = self.__detect_AC(self.__g(info['lam_g']))
        self.__log['nAC'].append(nAC)
        self.__log['idx_AC'].append(nAC)
        self.__log['nAS'].append(info['nAS'])

        return None

    def __detect_AC(self, lam_g_opt):

        # optimal active set
        if 'h' in lam_g_opt.keys():
            idx_opt = [k for k in range(self.__h.size1_out(0)-self.__nsc) if lam_g_opt['h',0][k] != 0]
            lam_g_ref = self.__g(self.__ref_du[self.__index])
            idx_ref  = [k for k in range(self.__h.size1_out(0)-self.__nsc) if lam_g_ref['h',0][k] != 0]

        else:
            idx_opt = []
            idx_ref = []

        # get number of active set changes
        nAC = len([k for k in idx_opt if k not in idx_ref])
        nAC += len([k for k in idx_ref if k not in idx_opt])

        return nAC, idx_opt

    def reset(self):

        self.__index = 0
        self.__initialize_log()
        self.__set_initial_guess()

        return None

    def __shift_initial_guess(self, w0, lam_g0):

        w_shifted = self.__w(0.0)
        lam_g_shifted = self.__g(0.0)
        lam_g_shifted['init'] = lam_g0['dyn',0]

        # shift states and controls
        for i in range(self.__N):

            # shift primal solution
            w_shifted['x',i] = w0['x',i+1]

            if i < self.__N-1:
                w_shifted['u',i] = w0['u',i+1]
                if 'us' in self.__vars:
                    w_shifted['us',i] = w0['us',i+1]
                if 'usc' in self.__vars:
                    w_shifted['usc',i] = w0['usc',i+1]

                # shift dual solution
                lam_g_shifted['dyn',i] = lam_g0['dyn', i+1]
                for constr in ['g','h']:
                    if constr in lam_g0.keys():
                        lam_g_shifted[constr,i] = lam_g0[constr, i+1]

        # copy final interval
        w_shifted['x', self.__N] = w_shifted['x', self.__N-1]
        w_shifted['u', self.__N-1] = w_shifted['u', self.__N-2]
        if 'us' in self.__vars:
            w_shifted['us', self.__N-1] = w_shifted['us', self.__N-2]
        if 'usc' in self.__vars:
            w_shifted['usc', self.__N-1] = w_shifted['usc', self.__N-2]

        lam_g_shifted['dyn',self.__N-1] = lam_g_shifted['dyn',self.__N-2]
        for constr in ['g','h']:
            if constr in lam_g0.keys():
                lam_g_shifted[constr,self.__N-1] = lam_g_shifted[constr,self.__N-2]
        lam_g_shifted['term'] = lam_g0['term']

        return w_shifted, lam_g_shifted

    def __set_initial_guess(self):

        # create initial guess at steady state
        wref = self.__wref(self.__ref[self.__index])
        w0   = self.__w(0.0)
        w0['x'] = wref['x']
        w0['u'] = wref['u']
        if 'us' in self.__vars:
            w0['us'] = wref['us']
        self.__w0 = w0

        # initial guess for multipliers
        self.__lam_g0 = self.__g(self.__ref_du[self.__index])

        return None

    @property
    def w(self):
        return self.__w

    @property
    def g_sol(self):
        return self.__g_sol

    @property
    def w_sol(self):
        return self.__w_sol

    @property
    def log(self):
        return self.__log

    @property
    def index(self):
        return self.__index