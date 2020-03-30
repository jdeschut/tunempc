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
""" Convexification code

:author: Jochem De Schutter

"""

import picos
import numpy as np
import tunempc.mtools as mtools
from functools import reduce
import tunempc.preprocessing as preprocessing
from tunempc.logger import Logger

def convexify(A, B, Q, R, N, C = None, opts = {'rho':1e-3, 'solver':'mosek','force': False}):

    """ Convexify the indefinite Hessian "H" of the system with the discrete time dynamics

    .. math::
        
        x_{k+1} = A x_k + B u_k

    so that the solution of the LQR problem based on the convexified
    Hessian "H + dH" yields the same trajectory as the LQR-solution
    of the indefinite problem.

    :param A: system matrix
    :param B: input matrix
    :param Q: weighting matrix Q (nx,nx)
    :param R: weighting matrix R (nu,nu)
    :param N: weighting matrix N (nx,nu)
    :param C: jacobian of active constraints at steady state (nc, nx+nu)
    :param opts: tuning options

    :return: Convexified Hessian supplement "dH".
    """

    # perform input checks
    arg = {**locals()}
    del arg['opts']

    # extract steady-state period
    period = len(arg['A'])
    
    Logger.logger.info('Convexify Hessians along {:d}-periodic steady state trajectory.'.format(period))

    if arg['C'] is None:
        del arg['C']
        Logger.logger.info('Convexifier called w/o active constraints at steady state')

    arg = preprocessing.input_checks(arg)

    # extract dimensions
    nx = arg['A'][0].shape[0]
    nu = arg['B'][0].shape[1]

    # check if hessian is already convex!
    min_eigval = list(map(lambda q,r,n: np.min(np.linalg.eigvals(mtools.buildHessian(q,r,n))), arg['Q'], arg['R'], arg['N']))
    if min(min_eigval) > 0:
        Logger.logger.info('Provided hessian(s) are already positive definite. No convexification needed!')
        return  np.zeros((nx+nu,nx+nu)), np.zeros((nx,nx)), np.zeros((nu,nu)), np.zeros((nx,nu))

    # solver verbosity
    if Logger.logger.getEffectiveLevel() < 20:
        opts['verbose'] = 1
    else:
        opts['verbose'] = 0

    Logger.logger.info('Construct SDP...')
    Logger.logger.info('')
    Logger.logger.info(50*'*')

    # perform autoscaling
    scaling = autoScaling(arg['Q'], arg['R'], arg['N'])

    # define model
    M = setUpModelPicos(**arg, constr = False)

    Logger.logger.info('Step 1: (\u03B7_F = 0), (\u03B7_T = 0)')

    # solve
    constraint_contribution = False
    M = solveSDP(M, opts)

    status, dHc, dQc, dRc, dNc = check_convergence(M, scaling, **arg, constr = constraint_contribution)

    if status in ['Optimal', 'Feasible']:

        Logger.logger.info('EQUIVALENCE TYPE A')
        Logger.logger.info(50*'*')

    if status == 'Infeasible' and 'C' in arg:
        
        Logger.logger.info(50*'*')
        Logger.logger.info('Step 2: (\u03B7_F = 1), (\u03B7_T = 0)')

        # create model with active constraint regularisation
        M = setUpModelPicos(**arg, rho = opts['rho'], constr = True)

        # solve again
        constraint_contribution = True
        M = solveSDP(M, opts)

        status, dHc, dQc, dRc, dNc = check_convergence(M, scaling, **arg, constr = constraint_contribution)

        if status in ['Optimal', 'Feasible']:

            Logger.logger.info('EQUIVALENCE TYPE B')
            Logger.logger.info(50*'*')

    if status == 'Infeasible':

        Logger.logger.warning('!! Strict dissipativity does not hold locally !!')
        Logger.logger.warning('!! The provided indefinite LQ MPC problem is not stabilising !!')
        Logger.logger.warning(50*'*')

        if opts['force']:

            Logger.logger.info('Step 3: (\u03B7_F = 1), (\u03B7_T = 1)')
            Logger.logger.info('Enforcing convexification...')
            raise ValueError('Step 3 not implemented yet.')
            Logger.logger.warning(50*'*')

        else:

            Logger.logger.warning('Consider operating the system at another orbit of different period p')
            Logger.logger.warning('Convexification and stabilization of the MPC scheme can be enforced by enabling "force"-flag.')
            Logger.logger.warning('In this case there are no guarantees of (local, first-order) equivalence.')
            raise ValueError('Convexification is not possible if the system is not optimally operated at the optimal orbit.')

    Logger.logger.info('')
    Logger.logger.info('Hessians convexified.')
    Logger.logger.info('')

    return dHc, dQc, dRc, dNc

def convexHessianSuppl(A, B, Q, R, N, dP, C = None, F = None):

    """ Construct the convexified Hessian Supplement

    :param A: system matrix
    :param B: input matrix
    :param Q: weighting matrix Q (nx,nx)
    :param R: weighting matrix R (nu,nu)
    :param N: weighting matrix N (nx,nu)
    :param dP: ...

    :return: convexified Hessian supplement
    """

    period = len(dP)
    nx = A[0].shape[0] # state dimension

    dHc, dQc, dRc, dNc = [], [], [], []

    for i in range(period):

        # unpack dP
        dP1 = dP[i]
        dP2 = dP[(i+1)%period]

        # convexify
        Qco = np.squeeze(np.dot(np.dot(A[i].T, dP2), A[i]) - dP1)
        Rco = np.dot(np.dot(B[i].T, dP2), B[i])
        Nco = np.dot(np.dot(B[i].T, dP2.T), A[i]).T
        Hco = mtools.buildHessian(Qco, Rco, Nco)

        if F:
            if C[i] is not None:
                nc = C[i].shape[0]
                Hco = Hco + np.dot(np.dot(C[i].T, np.diagflat(F[i])), C[i])

        # symmetrize
        dHc.append(mtools.symmetrize(Hco))
        dQc.append(dHc[i][:nx,:nx])
        dRc.append(dHc[i][nx:,nx:])
        dNc.append(dHc[i][:nx,nx:])

    return dHc, dQc, dRc, dNc

def setUpModelPicos(A, B, Q, R, N, C = None, rho = 1e-3, constr = True):

    """ Description
    :param A:
    :param B:
    :param Q:
    :param R:
    :param N:
    :param dP:
    :param C:
    :param constr:

    :rtype:
    """

    # extract data
    nx = A[0].shape[0]
    nu = B[0].shape[1]
    period = len(A)

    # create model
    M  = picos.Problem()

    # scaling
    scaling = autoScaling(Q,R,N)

    # model variables
    alpha = M.add_variable('alpha', 1, vtype='continuous')
    beta  = M.add_variable('beta', 1, vtype='continuous')
    dP = [M.add_variable('dP'+str(i), (nx,nx), vtype='symmetric') for i in range(period)]

    # alpha should be positive
    M.add_constraint(alpha > 1e-8)

    # add regularisation in null space of constraints
    if (C is not None) and (constr is True): # TODO check if this is correct
        F, t = [], []
        for i in range(period):
            if C[i] is not None:
                nc = C[i].shape[0] # number of active constraints at index
                F.append(M.add_variable('F'+str(i), (nc,1), vtype='continuous'))
                M.add_constraint(F[i] > 0)
            else:
                F.append(None)

    # objective
    obj = beta # minimize condition number
    if constr is True:
        for i in range(period):
            obj = picos.sum(obj, abs(rho*F[i]))

    M.set_objective('min', obj)

    # formulate convexified Hessian expression
    if not constr:
        HcE = [convexHessianExprPicos(Q, R, N, A, B, dP, alpha, scaling, index = i)  \
            for i in range(period)]
    else:
        HcE = [convexHessianExprPicos(Q, R, N, A, B, dP, alpha, scaling, C = C, F = F, index = i) \
             for i in range(period)]

    # formulate constraints
    for i in range(period):
        M.add_constraint((HcE[i] - np.eye(nx+nu))/scaling['alpha'] >> 0)
        M.add_constraint((scaling['beta']*beta * np.eye(nx+nu) - HcE[i])/scaling['alpha'] >> 0)

    return M

def convexHessianExprPicos(Q, R, N, A, B, dP, alpha, scaling, index, C = None, F = None):

    """ Construct the Picos symbolic expression of the convexified Hessian
    
    :param H: non-convex Hessian
    :param A: system matrix
    :param B: input matrix
    :param dP: ...

    :return: Picos Expression of the convexified Hessian
    """

    # set-up
    period = len(dP)

    H = scaling['alpha']*alpha*mtools.buildHessian(Q[index], R[index], N[index])

    A = A[index]
    B = B[index]

    if C is not None:
        CM = C[index]

    dP1 = scaling['dP']*dP[index]
    dP2 = scaling['dP']*dP[(index+1)%period]

    # convexify
    dQ = A.T*dP2*A - dP1
    dN = A.T*dP2*B
    dNT = B.T*dP2.T*A
    dR = B.T*dP2*B
    dH = (dQ & dN) // (dN.T & dR)

    # add constraints contribution
    if F is not None:
        if F[index] is not None:
            dH = dH + CM.T*picos.diag(scaling['F']*F[index])*CM

    return H+dH

def solveSDP(M, opts):

    Logger.logger.info('solving SDP...')

    M.solve(solver = opts['solver'], verbose = opts['verbose'])

    if M.status == 'optimal':
        Logger.logger.debug('SDP solution:')
        Logger.logger.debug('alpha: {}'.format(str(M.variables['alpha'].value)))
        Logger.logger.debug('beta: {}'.format(str((M.variables['beta'].value))))
    else:
        Logger.logger.debug('solution status: {} ...'.format(M.status))

    return M

def autoScaling(Q, R, N):

    """ Compute scaling factors for SDP variables, objective and constraints.
    """

    # build hessians
    H = [mtools.buildHessian(Q[k], R[k], N[k]) for k in range(len(Q))]

    # get mininmum absolute value of eigenvalues
    min_eig = 1e10
    max_eig = 0.0
    for k in range(len(H)):
        eigvalsk = np.abs(np.linalg.eigvals(H[k]))
        min_eig = min([min_eig, np.min(eigvalsk[np.nonzero(eigvalsk)])])
        max_eig = max([max_eig, np.max(eigvalsk[np.nonzero(eigvalsk)])])
    
    # maximum condition number
    max_cond = max_eig/min_eig
    min_eig = 1e-8 # SDP SCALING!!!
    max_cond = 1e8

    scaling = {
        'alpha': 1/min_eig,
        'beta': max_cond,
        'dP': 1/min_eig,
        'F': 1/min_eig,
        'T': 1/min_eig
    }
    
    return scaling

def check_convergence(M, scaling, A, B, Q, R, N, C = None, constr = False):

    # build convex hessian list
    dP = [scaling['dP']*np.array(M.variables['dP'+str(i)].value)/(scaling['alpha']*M.variables['alpha'].value) \
        for i in range(len(A))]

    if not constr:
        dHc, dQc, dRc, dNc = convexHessianSuppl( A, B, Q, R, N, dP)
    else:
        F = [scaling['F']*np.array(M.variables['F'+str(i)].value)/(scaling['alpha']*M.variables['alpha'].value) \
            for i in range(len(A))]
        dHc, dQc, dRc, dNc = convexHessianSuppl( A, B, Q, R, N, dP, C = C,  F =  F)

    # add hessian supplements
    Hc = [mtools.buildHessian(Q[k], R[k], N[k]) + dHc for k in range(len(dHc))]

    # compute eigenvalues
    min_eigenvalue = min([np.min(np.linalg.eigvals(Hk)) for Hk in Hc])
    max_eigenvalue = max([np.max(np.linalg.eigvals(Hk)) for Hk in Hc])
    max_cond       = max([np.linalg.cond(Hk) for Hk in Hc])[0]

    if min_eigenvalue > 1e-8:
        if M.status == 'optimal':
            status = 'Optimal'
        else:
            status = 'Feasible'
        Logger.logger.info('{} solution found.'.format(status))
        Logger.logger.info('Maximum condition number: {}'.format(max_cond))
        Logger.logger.info('Minimum eigenvalue: {}'.format(min_eigenvalue))
    else:
        status = 'Infeasible'
        Logger.logger.info('SDP solver status: {}'.format(M.status))
        Logger.logger.info('!! Problem infeasible !!')

    return status, dHc, dQc, dRc, dNc