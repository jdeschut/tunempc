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
""" Matrix operation tools.

:author: Jochem De Schutter

"""

import numpy as np
import casadi as ca
import casadi.tools as ct

def symmetrize(S):
    """Symmetrize matrix S
    """
    return (S + S.T)/2.0

def buildHessian(Q, R, N):
    """ Build the Hessian matrix based on submatrices.
    """
    return np.vstack((np.hstack((Q, N)),np.hstack((N.T, R))))

def tracking_cost(nw):
    """ Create tracking cost function for variables nw
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