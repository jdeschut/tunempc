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
Preprocessing routines of provided user data

:author: Jochem De Schutter
"""

import casadi as ca
import casadi.tools as ct
import itertools
import numpy as np
import collections
from tunempc.logger import Logger

def input_formatting(sys):

    # TODO: perform some tests?

    # system dimensions
    if type(sys['f']) == list:
        fsize = sys['f'][0]
    else:
        fsize = sys['f'] 
    nx = fsize.size1_in(0)
    nu = fsize.size1_in(1)
    sys['vars'] = collections.OrderedDict()
    sys['vars']['x'] = ca.MX.sym('x',nx)
    sys['vars']['u'] = ca.MX.sym('u',nu)

    # process path constraints
    if 'h' in sys:

        # detect and sort out nonlinear inequalities
        sys['g'], sys['h'] = detect_nonlinear_inequalities(sys['h'])

        if sys['g'].count(None) == len(sys['g']):
            del(sys['g'])
        else:
            # add slacks to system variables
            sys['vars']['us'] = []
            for k in range(len(sys['g'])):
                if sys['g'][k] is not None:
                    ns = sys['g'][k].size1_in(2)
                else:
                    ns = 0

            # TODO: allow for varying dimension of g    
            sys['vars']['us'] = ca.MX.sym('us',ns)

        # unpack lists for len == 1
        if len(sys['h']) == 1:
            sys['h'] = sys['h'][0]
            if 'g' in sys:
                sys['g'] = sys['g'][0]

    return sys

def detect_nonlinear_inequalities(h):

    # make constraints "time-varying"
    if type(h) is not list:
        h = [h]
    
    # initialize update inequality list
    h_new = []
    g_new = []

    # iterate over constraints
    for k in range(len(h)):
        x = ca.MX.sym('x', h[k].size1_in(0))
        u = ca.MX.sym('u', h[k].size1_in(1))
        h_expr = h[k](x,u)

        # sort constraints according to (non-)linearity
        h_nlin = []
        h_lin  = []
        for i in range(h_expr.shape[0]):
            if True in ca.which_depends(h_expr[i],ca.vertcat(x,u),2):
                h_nlin.append(h_expr[i])
            else:
                h_lin.append(h_expr[i])

        # update control vector (add slacks for nonlinear inequalities)
        if len(h_nlin) > 0:

            # function for nonlinear slacked inequalities
            s = ca.MX.sym('s', len(h_nlin))
            g_new.append(ca.Function('g',[x,u,s], [ca.vertcat(*h_nlin) - s]))

            # function for linear inequalities
            h_lin.append(s) # slacks > 0
            h_new.append(ca.Function('h', [x,u,s], [ca.vertcat(*h_lin)]))

        else:
            g_new.append(None)
            h_new.append(h[k])

    return g_new, h_new

def add_mpc_slacks(sys, lam_g, active_set, slack_flag = 'active'):

    """
    Add slacks to (all, active or none of) the linear inequality constraints
    in order to implement soft constraints in the MPC controller.
    """

    if ('h' not in sys) or (slack_flag == 'none'):
        return sys

    active_constraints = set(itertools.chain(*active_set))
    slack_condition = lambda x: (slack_flag == 'all') or \
            ((slack_flag == 'active') and (x in active_constraints))
    h_args = sys['vars'].values()
    h_expr = sys['h'](*h_args)

    slacks = list(map(slack_condition, range(h_expr.shape[0])))
    if sum(slacks) > 0:
        usc = ca.MX.sym('usc', sum(slacks))
        slack_cost = []
        h_slack = []
        j = 0
        for i in range(h_expr.shape[0]):
            if slacks[i]:
                h_slack.append(usc[j])
                slack_cost.append(1e3*np.max(-ca.vertcat(*lam_g['h',:,i]).full()))
                j = j+1
            else:
                h_slack.append(0.0)

        h_new = ca.vertcat(h_expr + ca.vertcat(*h_slack), usc)
        sys['h'] = ca.Function('h', [*h_args, usc], [h_new])
        sys['scost'] = ct.vertcat(*slack_cost)
        sys['vars']['usc'] = usc

    return sys

def input_checks(arg):

    """
    Input checks for provided convexification matrices A, B, Q, R, N, G, (C).

    - Check if all provided matrices are of same type (list vs. single matrix)
    - Check if size of matrices is consistent troughout time
    - Return matrices as lists
    """

    msg1 = "Input arguments should be of same type!"
    assert (all(type(argument)==type(arg['A']) for key, argument in arg.items())), msg1

    if type(arg['A']) == list:
        msg2 = "Input data lists should have same length!"
        assert (all(len(argument)==len(arg['A']) for key, argument in arg.items())), msg2
    else:
        for key in list(arg.keys()):
            arg[key] = [arg[key]]

    # check matrix size consistency
    msg3 = "Data matrices should have same size along trajectory."
    for key, argument in arg.items():
        if key != 'C':
            assert (all(mat.shape==argument[0].shape for mat in argument)), msg3

    # TODO: check A and B, Q, R, N on consistency

    return arg