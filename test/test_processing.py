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
Test routines for preprocessing module

:author: Jochem De Schutter
"""

import tunempc.preprocessing as preprocessing
import casadi as ca
import collections

def test_input_formatting_no_constraints():

    """ Test formatting of user provided input w/o constraints.
    """

    x = ca.MX.sym('x',1)
    u = ca.MX.sym('u',2)

    # case of no constraints
    sys = {}
    sys['F'] = ca.Function('F',[x,u],[x])
    sys = preprocessing.input_formatting(sys)

    assert 'h' not in sys, 'non-existent lin. inequalites are being created'
    assert 'g' not in sys, 'non-existent nonlin. inequalities are being created'
    assert 'vars'  in sys, 'system variables are not being created'
    assert sys['vars']['x'].shape == x.shape, 'x-variable dimensions are incorrect'
    assert sys['vars']['u'].shape == u.shape, 'u-variable dimensions are incorrect'

def test_input_formatting_lin_constraints():

    """ Test formatting of user provided input with linear constraints.
    """

    x = ca.MX.sym('x',1)
    u = ca.MX.sym('u',2)

    # case of linear constraints
    sys = {}
    sys['F'] = ca.Function('F',[x,u],[x])
    h = ca.Function('h',[x,u],[ca.vertcat(x+u[0], u[1])])
    sys['h'] = h
    sys = preprocessing.input_formatting(sys)

    assert 'h' in sys, 'lin. inequalites are being removed'
    assert sys['h'].size1_out(0) == 2, 'lin inequalities dimension is altered incorrectly'
    assert 'g' not in sys, 'non-existent nonlin. inequalities are being created'
    assert 'vars'  in sys, 'system variables are not being created'
    assert sys['vars']['x'].shape == x.shape, 'x-variable dimensions are incorrect'
    assert sys['vars']['u'].shape == u.shape, 'u-variable dimensions are incorrect'


def test_input_formatting_mixed_constraints():

    """ Test formatting of user provided input with linear constraints.
    """

    x = ca.MX.sym('x',1)
    u = ca.MX.sym('u',2)

    # case of mixed constraints
    sys = {}
    sys['F'] = ca.Function('F',[x,u],[x])
    h = ca.Function('h',[x,u],[ca.vertcat(x+u[0], u[1], x**2*u[0])])
    sys['h'] = h
    sys = preprocessing.input_formatting(sys)

    assert 'h' in sys, 'lin. inequalites are being removed'
    assert sys['h'].size1_out(0) == 3, 'lin inequalities have wrong dimension'
    assert 'g' in sys, 'nonlin. inequalities are not being created'
    assert sys['g'].size1_out(0) == 1, 'nonlin inequalities have wrong dimension'
    assert 'vars'  in sys, 'system variables are not being created'
    assert sys['vars']['x'].shape == x.shape, 'x-variable dimensions are incorrect'
    assert sys['vars']['u'].shape == u.shape, 'u-variable dimensions are incorrect'
    assert sys['vars']['us'].shape == (1,1), 'us-variable dimensions are incorrect'

    x_num = 1.0
    u_num = ca.vertcat(0.0, 2.0)
    us_num = ca.vertcat(3.0)
    h_eval = sys['h'](x_num, u_num, us_num).full()
    g_eval = sys['g'](x_num, u_num, us_num).full()
    assert ca.vertsplit(h_eval) == [1.0, 2.0, 3.0]
    assert ca.vertsplit(g_eval) == [-3.0]

def test_add_mpc_slacks_no_constraints():

    """ Test automatic soft constraint generation if no constraints are present.
    """

    x = ca.MX.sym('x',1)
    u = ca.MX.sym('u',2)

    # case of no constraints
    sys = {}
    sys['F'] = ca.Function('F',[x,u],[x])
    sys['vars'] = collections.OrderedDict()
    sys['vars']['x'] = x
    sys['vars']['u'] = u
    sys = preprocessing.add_mpc_slacks(sys, None, None, slack_flag = 'active')
    
    assert 'h' not in sys, 'non-existent lin. inequalities are being created'
    assert 'usc' not in sys['vars'], 'non-existent slack variables are being created'

def test_add_mpc_slacks_no_active_constraints():

    """ Test automatic soft constraint generation if no active constraints are present.
    """

    x = ca.MX.sym('x',1)
    u = ca.MX.sym('u',2)

    # case of no active constraints
    sys = {}
    sys['F'] = ca.Function('F',[x,u],[x])
    h = ca.Function('h',[x,u],[ca.vertcat(x+u[0], u[1])])
    sys['h'] = h
    sys['vars'] = collections.OrderedDict()
    sys['vars']['x'] = x
    sys['vars']['u'] = u
    active_set = [[],[]]
    sys = preprocessing.add_mpc_slacks(sys, None, active_set, slack_flag = 'active')
    
    assert 'h' in sys, 'lin. inequalities are being removed'
    assert 'usc' not in sys['vars'], 'non-existent slack variables are being created'

def test_add_mpc_slacks_active_constraints():

    """ Test automatic soft constraint generation if active constraints are present.
    """

    x = ca.MX.sym('x',1)
    u = ca.MX.sym('u',2)

    # case of no active constraints
    sys = {}
    sys['F'] = ca.Function('F',[x,u],[x])
    h = ca.Function('h',[x,u],[ca.vertcat(x+u[0], u[1])])
    sys['h'] = h
    sys['vars'] = collections.OrderedDict()
    sys['vars']['x'] = x
    sys['vars']['u'] = u

    lam_g_struct = ca.tools.struct_symMX([
        ca.tools.entry('h', shape = (2,1), repeat = 2)
    ])
    lam_g = lam_g_struct(0.0)
    lam_g['h',0,0] = -5.0
    active_set = [[0],[]]
    sys = preprocessing.add_mpc_slacks(sys, lam_g, active_set, slack_flag = 'active')
    
    assert 'h' in sys, 'lin. inequalities are being removed'
    assert 'usc' in sys['vars'], 'slack variables are not being created'
    assert sys['vars']['usc'].shape[0] == 1, 'slack variables are not being created'
    assert 'scost' in sys, 'slack cost is not being created'

    x_num = 1.0
    u_num = ca.vertcat(0.0, 2.0)
    usc_num = ca.vertcat(3.0)
    h_eval = sys['h'](x_num, u_num, usc_num).full()
    scost_eval = sys['scost'](usc_num).full()
    assert ca.vertsplit(h_eval) == [4.0, 2.0, 3.0]
    assert ca.vertsplit(scost_eval) == [150.0]