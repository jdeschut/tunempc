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
MPC routines for periodic (standard/economic) reference tracking.

:author: Jochem De Schutter
"""

import casadi.tools as ct
import casadi as ca
import numpy as np
import itertools
import copy
import collections
import tunempc.sqp_method as sqp_method
from tunempc.logger import Logger

class Pmpc(object):

    def __init__(self, N, sys, cost, wref = None, tuning = None, lam_g_ref = None, sensitivities = None, options = {}):
        
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

        # store system sensitivities around steady state
        self.__S = sys['S']

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
        self.__jac_p_operator = ca.Function(
            'jac_p',
            [sys['vars']['x']],
            [ca.jacobian(self.__p_operator(sys['vars']['x']),sys['vars']['x'])]
        )
        self.__S = sensitivities

        # construct MPC solver
        self.__construct_solver()

        # periodic indexing
        self.__index = 0
        self.__index_acados = 0

        # create periodic reference
        assert wref   != None, 'Provide reference trajectory!'
        self.__create_reference(wref, tuning, lam_g_ref)

        # initialize log
        self.__initialize_log()

        # initialize acados solvers
        self.__acados_ocp_solver = None
        self.__acados_integrator = None

        # solver initial guess
        self.__set_initial_guess()

        return None

    def __default_options(self):

        # default options
        opts = {
            'hessian_approximation': 'exact',
            'ipopt_presolve': False,
            'max_iter': 2000,
            'p_operator': ca.Function('p_operator',[self.__vars['x']],[self.__vars['x']]),
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
        self.__nx_term = self.__p_operator.size1_out(0)

        # create general constraints structure
        g_struct = ct.struct_symMX([
            ct.entry('init', shape = (self.__nx,1)),
            constraints_entry,
            ct.entry('term', shape = (self.__nx_term,1))
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

        term_constraint = self.__p_operator(w['x',-1] - xN)

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
            if Logger.logger.getEffectiveLevel() > 10:
                opts['ipopt']['print_level'] = 0
                opts['print_time'] = 0
                opts['ipopt']['sb'] = 'yes'
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

    def step_acados(self, x0):

        # reset periodic indexing if necessary
        self.__index_acados = self.__index_acados%self.__Nref

        # format x0
        x0 = np.squeeze(x0.full())

        # update NLP parameters
        self.__acados_ocp_solver.set(0, "lbx", x0)
        self.__acados_ocp_solver.set(0, "ubx", x0)

        # update reference and tuning matrices
        self.__set_acados_reference()

        # solve
        status = self.__acados_ocp_solver.solve()

        # timings
        # np.append(self.__acados_times, self.__acados_ocp_solver.get_stats("time_tot"))
        print("acados timings: total: ", self.__acados_ocp_solver.get_stats("time_tot"), \
            " lin: ", self.__acados_ocp_solver.get_stats("time_lin"), \
            " sim: ", self.__acados_ocp_solver.get_stats("time_sim"), " qp: ", \
                 self.__acados_ocp_solver.get_stats("time_qp"))

        # if status != 0:
        #     raise Exception('acados solver returned status {}. Exiting.'.format(status))

        # save solution
        self.__w_sol_acados = self.__w(0.0)
        for i in range(self.__N):
            self.__w_sol_acados['x',i] = self.__acados_ocp_solver.get(i,"x")
            self.__w_sol_acados['u',i] = self.__acados_ocp_solver.get(i,"u")[:self.__nu]
            if 'us' in self.__vars:
                self.__w_sol_acados['us',i] = self.__acados_ocp_solver.get(i,"u")[self.__nu:]
        self.__w_sol_acados['x',self.__N] = self.__acados_ocp_solver.get(self.__N,"x")
        self.__extract_acados_solver_stats()

        # feedback policy
        u0 = self.__acados_ocp_solver.get(0, "u")[:self.__nu]

        # update initial guess
        self.__shift_initial_guess_acados()

        # shift index
        self.__index_acados += 1

        return u0

    def generate(self, dae, name = 'tunempc', opts = {}):

        """ Create embeddable NLP solver
        """

        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver

        # extract dimensions
        nx = self.__nx
        nu = self.__nu + self.__ns# treat slacks as pseudo-controls 

        # extract reference
        ref = self.__ref
        xref = np.squeeze(self.__ref[0][:nx], axis = 1)
        uref = np.squeeze(self.__ref[0][nx: nx + nu], axis = 1)

        # create acados model
        model = AcadosModel()
        model.x = ca.MX.sym('x',nx)
        model.u = ca.MX.sym('u',nu)
        model.p = []
        model.name = name

        # detect input type
        n_in = dae.n_in()
        if n_in == 2:

            # xdot = f(x, u)
            if 'integrator_type' in opts:
                if opts['integrator_type'] in ['IRK','GNSF']:
                    xdot = ca.MX.sym('xdot', nx)
                    model.xdot = xdot
                    model.f_impl_expr = xdot - dae(model.x, model.u[:self.__nu])
                    model.f_expl_expr = xdot
                elif opts['integrator_type'] == 'ERK':
                    model.f_expl_expr = dae(model.x, model.u[:self.__nu])
            else:
                raise ValueError('Provide numerical integrator type!')

        else:

            xdot = ca.MX.sym('xdot', nx)
            model.xdot = xdot
            model.f_expl_expr = xdot

            if n_in == 3:

                # f(xdot, x, u) = 0
                model.f_impl_expr = dae(xdot, model.x, model.u[:self.__nu])

            elif n_in == 4:

                # f(xdot, x, u, z) = 0 
                nz = dae.size1_in(3)
                z = ca.MX.sym('z', nz)
                model.z = z
                model.f_impl_expr = dae(xdot, model.x, model.u[:self.__nu], z)
            else:
                raise ValueError('Invalid number of inputs for system dynamics function.')

        if self.__gnl is not None:
            model.con_h_expr = self.__gnl(model.x, model.u[:self.__nu], model.u[self.__nu:])

        if self.__type == 'economic':
            model.cost_expr_ext_cost = self.__cost(model.x, model.u[:self.__nu])/opts['tf']*self.__N


        # create acados ocp
        ocp = AcadosOcp()
        ocp.model = model
        ny = nx + nu
        ny_e = nx

        if 'integrator_type' in opts and opts['integrator_type'] == 'GNSF':
            from acados_template import acados_dae_model_json_dump
            import os
            acados_dae_model_json_dump(model)
            # Set up Octave to be able to run the following:
            ## if using a virtual python env, the following lines can be added to the env/bin/activate script:
            # export OCTAVE_PATH=$OCTAVE_PATH:$ACADOS_INSTALL_DIR/external/casadi-octave
            # export OCTAVE_PATH=$OCTAVE_PATH:$ACADOS_INSTALL_DIR/interfaces/acados_matlab_octave/
            # export OCTAVE_PATH=$OCTAVE_PATH:$ACADOS_INSTALL_DIR/interfaces/acados_matlab_octave/acados_template_mex/
            # echo
            # echo "OCTAVE_PATH=$OCTAVE_PATH"
            status = os.system(
                "octave --eval \"convert({})\"".format("\'"+model.name+"_acados_dae.json\'")
            )
            # load gnsf from json
            with open(model.name + '_gnsf_functions.json', 'r') as f:
                import json
                gnsf_dict = json.load(f)
            ocp.gnsf_model = gnsf_dict

        # set horizon length
        ocp.dims.N = self.__N

        # set cost module
        if self.__type == 'economic':

            # set cost function type to external (provided in model)
            ocp.cost.cost_type = 'EXTERNAL'
        else:

            # set weighting matrices
            if self.__type == 'tracking':
                ocp.cost.W = self.__Href[0][0]

            # set-up linear least squares cost
            ocp.cost.cost_type = 'LINEAR_LS'
            ocp.cost.W_e = np.zeros((nx, nx))
            ocp.cost.Vx = np.zeros((ny, nx))
            ocp.cost.Vx[:nx,:nx] = np.eye(nx)
            Vu = np.zeros((ny, nu))
            Vu[nx:,:] = np.eye(nu)
            ocp.cost.Vu = Vu
            ocp.cost.Vx_e = np.eye(nx)
            ocp.cost.yref  = np.squeeze(
                ca.vertcat(xref,uref).full() - \
                ct.mtimes(np.linalg.inv(ocp.cost.W),self.__qref[0][0].T).full(), # gradient term
                axis = 1
                )
            ocp.cost.yref_e = np.zeros((ny_e, ))
            if n_in == 4: # DAE flag
                ocp.cost.Vz = np.zeros((ny,nz))

        if 'custom_hessian' in opts:
            self.__custom_hessian = opts['custom_hessian']

        # initial condition
        ocp.constraints.x0 = xref

        # set inequality constraints
        ocp.constraints.constr_type = 'BGH'
        if self.__S['C'] is not None:
            C = self.__S['C'][0][:,:nx]
            D = self.__S['C'][0][:,nx:]
            lg = -self.__S['e'][0] + ct.mtimes(C,xref).full() + ct.mtimes(D,uref).full()
            ug = 1e8 - self.__S['e'][0] + ct.mtimes(C,xref).full() + ct.mtimes(D,uref).full()
            ocp.constraints.lg = np.squeeze(lg, axis = 1)
            ocp.constraints.ug = np.squeeze(ug, axis = 1)
            ocp.constraints.C  = C
            ocp.constraints.D  = D
            
            if 'usc' in self.__vars:
                if 'us' in self.__vars:
                    arg = [self.__vars['x'],self.__vars['u'], self.__vars['us'],self.__vars['usc']]
                else:
                    arg = [self.__vars['x'],self.__vars['u'],self.__vars['usc']]
                Jsg = ca.Function(
                    'Jsg',
                    [self.__vars['usc']],
                    [ca.jacobian(
                        self.__h(*arg),
                        self.__vars['usc']
                    )]
                )(0.0)
                self.__Jsg = Jsg.full()[:-self.__nsc,:]
                ocp.constraints.Jsg = self.__Jsg
                ocp.cost.Zl = np.zeros((self.__nsc,))
                ocp.cost.Zu = np.zeros((self.__nsc,))
                ocp.cost.zl = np.squeeze(self.__scost.full(), axis = 1)
                ocp.cost.zu = np.squeeze(self.__scost.full(), axis = 1)

        # set nonlinear equality constraints
        if self.__gnl is not None:
            ocp.constraints.lh = np.zeros(self.__ns,)
            ocp.constraints.uh = np.zeros(self.__ns,)

        # terminal constraint:
        x_term = self.__p_operator(model.x)
        Jbx = ca.Function('Jbx',[model.x], [ca.jacobian(x_term, model.x)])(0.0)
        ocp.constraints.Jbx_e = Jbx.full()
        ocp.constraints.lbx_e = np.squeeze(self.__p_operator(xref).full(), axis = 1)
        ocp.constraints.ubx_e = np.squeeze(self.__p_operator(xref).full(), axis = 1)

        for option in list(opts.keys()):
            if hasattr(ocp.solver_options, option):
                setattr(ocp.solver_options, option, opts[option])

        self.__acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
        self.__acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')

        # set initial guess
        self.__set_acados_initial_guess()

        return self.__acados_ocp_solver, self.__acados_integrator

    def __create_reference(self, wref, tuning, lam_g_ref):

        """ Create periodic reference and tuning data.
        """

        # period of reference
        self.__Nref = len(wref['u'])

        # create reference and tuning sequence
        # for each starting point in period
        ref_pr = []
        ref_du = []
        ref_du_struct = []
        H   = []
        q   = []

        for k in range(self.__Nref):
            
            # reference primal solution
            refk = []
            for j in range(self.__N):

                refk += [
                    wref['x', (k+j)%self.__Nref],
                    wref['u', (k+j)%self.__Nref]
                ]

                if 'us' in self.__vars:
                    refk += [wref['us', (k+j)%self.__Nref]]
            
            refk.append(wref['x', (k+self.__N)%self.__Nref])

            # reference dual solution
            lamgk = self.__g(0.0)
            lamgk['init'] = -lam_g_ref['dyn',(k-1)%self.__Nref]
            for j in range(self.__N):
                lamgk['dyn',j] = lam_g_ref['dyn',(k+j)%self.__Nref]
                if 'g' in list(lamgk.keys()):
                    lamgk['g',j] = lam_g_ref['g', (k+j)%self.__Nref]
                if 'h' in list(lamgk.keys()):
                    lam_h = [lam_g_ref['h', (k+j)%self.__Nref]]
                    if 'usc' in self.__vars:
                        lam_h += [-self.__scost] # TODO not entirely correct

                    lamgk['h',j] = ct.vertcat(*lam_h)
            lamgk['term'] = self.__p_operator(lam_g_ref['dyn',(k+self.__N-1)%self.__Nref])

            # adjust dual solution of terminal constraint is projected
            if self.__nx_term != self.__nx:

                # find new terminal multiplier
                A_m = []
                b_m = []
                A_factor = ca.DM.eye(self.__nx)
                for j in range(self.__N):
                    A_m.append(ct.mtimes(
                        ct.mtimes(self.__S['B'][(self.__N-j-1)%self.__Nref].T, A_factor),
                        self.__jac_p_operator(ca.DM.ones(self.__nx,1)).T
                        )
                    )
                    b_m.append(ct.mtimes(
                        ct.mtimes(self.__S['B'][(self.__N-j-1)%self.__Nref].T, A_factor),
                        lam_g_ref['dyn',(k+self.__N-1)%self.__Nref]
                        )
                    )
                    A_factor = ct.mtimes(self.__S['A'][(self.__N-j-1)%self.__Nref].T, A_factor)
                A_m = ct.vertcat(*A_m)
                b_m = ct.vertcat(*b_m)
                LI_indeces = [] # indeces of first full rank number linearly independent rows
                R0 = 0
                for i in range(A_m.shape[0]):
                    R = np.linalg.matrix_rank(A_m[LI_indeces+[i],:])
                    if R > R0:
                        LI_indeces.append(i)
                        R0 = R
                lamgk['term'] = ca.solve(A_m[LI_indeces,:], b_m[LI_indeces,:])

                # recursively update dynamics multipliers
                delta_lam = - lam_g_ref['dyn',(k+self.__N-1)%self.__Nref] + ct.mtimes(
                    self.__jac_p_operator(ca.DM.ones(self.__nx,1)).T,
                    lamgk['term']
                )
                lamgk['dyn',self.__N-1] += delta_lam
                for j in range(1,self.__N+1):
                    delta_lam = ct.mtimes(
                        self.__S['A'][(self.__N-j)%self.__Nref].T,
                        delta_lam
                    )
                    if j < self.__N:
                        lamgk['dyn', self.__N-1-j] += delta_lam
                    else:
                        lamgk['init'] += -delta_lam

            ref_pr.append(ct.vertcat(*refk))
            ref_du.append(lamgk.cat)
            ref_du_struct.append(lamgk)

            if tuning is not None:
                H.append([tuning['H'][(k+j)%self.__Nref] for j in range(self.__N)])
                q.append([tuning['q'][(k+j)%self.__Nref] for j in range(self.__N)])

        self.__ref    = ref_pr
        self.__ref_du = ref_du
        self.__ref_du_struct = ref_du_struct
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

        self.__log_acados = {
            'time_tot': [],
            'time_lin': [],
            'time_sim': [],
            'time_qp': [],
            'sqp_iter': [],
            'time_reg': [],
            'time_qp_xcond': [],
            'time_qp_solver_call': [],
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

    def __extract_acados_solver_stats(self):

        for key in list(self.__log_acados.keys()):
            self.__log_acados[key].append(self.__acados_ocp_solver.get_stats(key))

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
        self.__index_acados = 0
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

    def __shift_initial_guess_acados(self):

        for i in range(self.__N):
            x_prev = np.squeeze(self.__w_sol_acados['x',i+1].full(), axis = 1)
            self.__acados_ocp_solver.set(i, "x", x_prev)
            if i < self.__N-1:
                u_prev = np.squeeze(self.__w_sol_acados['u',i+1].full(), axis = 1)
                if 'us' in self.__vars:
                    u_prev = np.squeeze(ct.vertcat(u_prev, self.__w_sol_acados['us',i+1]).full(), axis = 1)
                self.__acados_ocp_solver.set(i, "u", u_prev)

        # initial guess in terminal stage on periodic trajectory
        idx = (self.__index_acados+self.__N)%self.__Nref

        # reference
        xref = np.squeeze(self.__ref[(idx+1)%self.__Nref][:self.__nx], axis = 1)
        uref = np.squeeze(self.__ref[idx][self.__nx: self.__nx + self.__nu + self.__ns], axis = 1)
        self.__acados_ocp_solver.set(self.__N, "x", xref)
        self.__acados_ocp_solver.set(self.__N-1, "u", uref)

        return None

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

        # acados solver initialization at reference
        if self.__acados_ocp_solver is not None:
            self.__set_acados_initial_guess()

        return None

    def __set_acados_reference(self):

        for i in range(self.__N):

            # periodic index
            idx = (self.__index_acados+i)%self.__Nref

            # reference
            xref = np.squeeze(self.__ref[idx][:self.__nx], axis = 1)
            uref = np.squeeze(self.__ref[idx][self.__nx: self.__nx + self.__nu + self.__ns], axis = 1)

            if self.__type == 'tracking':

                # construct output reference with gradient term
                yref = np.squeeze(
                    ca.vertcat(xref,uref).full() - \
                    ct.mtimes(
                        np.linalg.inv(self.__Href[idx][0]), # inverse of weighting matrix
                        self.__qref[idx][0].T).full(), # gradient term
                    axis = 1
                    )
                self.__acados_ocp_solver.set(i, 'yref', yref)

                # update tuning matrix
                self.__acados_ocp_solver.cost_set(i, 'W', self.__Href[idx][0])

            # set custom hessians if applicable
            if self.__acados_ocp_solver.acados_ocp.solver_options.ext_cost_custom_hessian:
                test = self.__custom_hessian[idx]
                self.__acados_ocp_solver.cost_set(i, "cost_custom_hess", self.__custom_hessian[idx])

            # update constraint bounds
            if self.__h is not None:
                C = self.__S['C'][idx][:,:self.__nx]
                D = self.__S['C'][idx][:,self.__nx:]
                lg = -self.__S['e'][idx] + ct.mtimes(C,xref).full() + ct.mtimes(D,uref).full()
                ug = 1e8 - self.__S['e'][idx] + ct.mtimes(C,xref).full() + ct.mtimes(D,uref).full()

                # remove constraints that depend on states only from first shooting node
                if i == 0:
                    for k in range(D.shape[0]):
                        if np.sum(D[k,:]) == 0.0:
                            lg[k] += -1e8
                    # TODO: for nonlinear constraints!!!
                self.__acados_ocp_solver.constraints_set(i, 'lg', np.squeeze(lg, axis = 1))
                self.__acados_ocp_solver.constraints_set(i, 'ug', np.squeeze(ug, axis = 1))

        # update terminal constraint
        idx = (self.__index_acados+self.__N)%self.__Nref
        x_term = np.squeeze(self.__p_operator(self.__ref[idx][:self.__nx]), axis = 1)
        self.__acados_ocp_solver.set(self.__N, 'lbx', x_term)
        self.__acados_ocp_solver.set(self.__N, 'ubx', x_term)

        return None

    def __set_acados_initial_guess(self):

        # dual reference solution
        ref_dual = self.__ref_du_struct[self.__index_acados%self.__Nref]

        for i in range(self.__N):

            # periodic index
            idx = (self.__index_acados+i)%self.__Nref

            # initialize at reference
            xref = np.squeeze(self.__ref[idx][:self.__nx], axis = 1)
            uref = np.squeeze(self.__ref[idx][self.__nx: self.__nx + self.__nu + self.__ns], axis = 1)

            # set initial guess
            self.__acados_ocp_solver.set(i, "x", xref)
            self.__acados_ocp_solver.set(i, "u", uref)

            # set dual initial guess
            self.__acados_ocp_solver.set(i, "pi", np.squeeze(ref_dual['dyn',i].full()))

            # the inequalities are internally organized in the following order:
            # [ lbu lbx lg lh ubu ubx ug uh ]
            lam_h = []
            t = []
            if i == 0:
                lam_x0 = copy.deepcopy(ref_dual['init'])
                if 'h' in list(ref_dual.keys()):
                    lam_lh0 = -ref_dual['h',i][:ref_dual['h',i].shape[0]-self.__nsc]
                    t_lh0 = copy.deepcopy(self.__S['e'][idx%self.__Nref])
                    if i == 0:
                        # set unused constraints at i=0 to be inactive
                        # TODO: for nonlinear constraints!!!
                        C = self.__S['C'][idx][:,:self.__nx]
                        D = self.__S['C'][idx][:,self.__nx:]
                        for k in range(D.shape[0]):
                            if np.sum(D[k,:]) == 0.0:
                                lam_x0 += - ct.mtimes(lam_lh0[k], C[k,:])
                                lam_lh0[k] = 0.0
                                t_lh0[k] += 1e8
                lam_lx0 = - copy.deepcopy(lam_x0)
                for k in range(self.__nx):
                    if lam_lx0[k] < 0.0:
                        lam_lx0[k] = 0.0 # assign multiplier to upper bound
                lam_h.append(lam_lx0)  # lbx_0
                t.append(np.zeros((self.__nx,)))
            if 'h' in list(ref_dual.keys()):
                if i == 0:
                    lam_lh = lam_lh0
                    t_lh = t_lh0
                else:
                    lam_lh = -ref_dual['h',i][:ref_dual['h',i].shape[0]-self.__nsc]
                    t_lh = copy.deepcopy(self.__S['e'][idx%self.__Nref])
                lam_h.append(lam_lh) # lg
                t.append(t_lh)
            if 'g' in list(ref_dual.keys()):
                lam_h.append(ref_dual['g',i]) # lh
                t.append(np.zeros((ref_dual['g',i].shape[0],)))
            if i == 0:
                lam_ux0 = copy.deepcopy(lam_x0)
                for k in range(self.__nx):
                    if lam_ux0[k] < 0.0:
                        lam_ux0[k] = 0.0 # assign multiplier to lower bound
                lam_h.append(lam_ux0)  # ubx_0
                t.append(np.zeros((self.__nx,)))
            if 'h' in list(ref_dual.keys()):
                lam_h.append(np.zeros((ref_dual['h',i].shape[0]- self.__nsc,))) # ug
                t.append(1e8*np.ones((ref_dual['h',i].shape[0]- self.__nsc,1))-self.__S['e'][idx])
            if 'g' in list(ref_dual.keys()):
                lam_h.append(np.zeros((ref_dual['g',i].shape[0],))) # uh
                t.append(np.zeros((ref_dual['g',i].shape[0],)))
            if self.__nsc > 0:
                lam_sl = self.__scost - ct.mtimes(lam_lh.T,self.__Jsg).T
                lam_h.append(lam_sl) # ls
                lam_h.append(self.__scost) # us
                t.append(np.zeros((self.__nsc,))) # slg > 0
                t.append(np.zeros((self.__nsc,))) # sug > 0
            if len(lam_h) != 0:
                self.__acados_ocp_solver.set(i, "lam", np.squeeze(ct.vertcat(*lam_h).full()))
                self.__acados_ocp_solver.set(i, "t", np.squeeze(ct.vertcat(*t).full()))

        # terminal state
        idx = (self.__index_acados+self.__N)%self.__Nref
        xref = np.squeeze(self.__ref[idx][:self.__nx], axis = 1)
        self.__acados_ocp_solver.set(self.__N, "x", xref)

        # terminal multipliers
        lam_lterm = -ref_dual['term']
        lam_uterm = np.zeros((ref_dual['term'].shape[0],))
        for k in range(lam_lterm.shape[0]):
            if lam_lterm[k] < 0.0:
                lam_uterm[k] = -lam_lterm[k]
                lam_lterm[k] = 0.0
        lam_term = np.squeeze(ct.vertcat(lam_lterm,lam_uterm).full())
        self.__acados_ocp_solver.set(self.__N, "lam", lam_term)

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
    def log_acados(self):
        return self.__log_acados

    @property
    def index(self):
        return self.__index

    @property
    def acados_ocp_solver(self):
        return self.__acados_ocp_solver

    @property
    def acados_integrator(self):
        return self.__acados_integrator

    @property
    def w_sol_acados(self):
        return self.__w_sol_acados
